from typing import (
    Any,
    Optional,
    TYPE_CHECKING,
    List,
)
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import uuid

from encapsulation.data_model.orm_models import FileMetadata
from encapsulation.data_model.orm_models import FileStatus

from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.core.file_management.storage.file_storage import FileStorageConfig

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails"""
    pass


class StorageOperationError(Exception):
    """Raised when storage operation fails"""
    pass


class FileStorage(AbstractModule):
    """
    Core file storage interface for RAG system.

    Provides high-level file upload, validation, and storage operations
    with coordination between blob storage and metadata storage.

    Key features:
    - File validation and metadata verification
    - Automatic cleanup on validation failures
    - Upload session tracking
    - Comprehensive error handling and reporting

    Architecture:
        Application Layer -> FileStorage (Core) -> Blob Storage + Metadata Storage

    Dependencies:
        blob_store: FileDB implementation (e.g., LocalDB, MinIODB)
        metadata_store: RelationalDB implementation (e.g., PostgreSQLDB)
    """

    def __init__(self, config:"FileStorageConfig"):
        """Initialize FileStorage with eager blob and metadata store creation"""
        super().__init__(config)
        # Build stores directly (no intermediate data_store layer)
        self.blob_store = config.file_db_config.build()
        self.metadata_store = config.relational_db_config.build()

    def _generate_upload_session_id(self) -> str:
        """Generate unique upload session ID"""
        return str(uuid.uuid4())

    def _generate_file_id(self) -> str:
        """Generate unique file ID"""
        return str(uuid.uuid4())

    def _generate_blob_key(self, file_id: str, filename: str) -> str:
        """Generate blob storage key from file ID and filename"""
        # Create hierarchical key: files/{first-2-chars-of-id}/{file-id}/{filename}
        prefix = file_id[:2]
        return f"files/{prefix}/{file_id}/{filename}"

    def _validate_file_upload(
        self,
        filename: str,
        file_data: bytes
    ) -> None:
        """Validate file upload parameters"""
        if not filename or not filename.strip():
            raise FileValidationError("Filename cannot be empty")

        if not file_data:
            raise FileValidationError("file_data must be provided")

        # Add more validation rules as needed
        max_filename_length = 255
        if len(filename) > max_filename_length:
            raise FileValidationError(f"Filename too long (max {max_filename_length} characters)")

        # Validate file size
        max_file_size = 100 * 1024 * 1024  # 100MB default limit
        if len(file_data) > max_file_size:
            raise FileValidationError(f"File too large (max {max_file_size} bytes)")

    def _validate_stored_file(self, file_id: str, **kwargs: Any) -> bool:
        """Validate that file was stored successfully"""
        try:
            # Get file metadata
            metadata = self.metadata_store.get_file_metadata(file_id, **kwargs)
            if not metadata:
                logger.warning(f"File metadata not found for validation: {file_id}")
                return False

            # Check if blob exists
            blob_exists = self.blob_store.exists(metadata.blob_key, **kwargs)
            if not blob_exists:
                logger.error(f"Blob validation failed - blob not found: {metadata.blob_key}")
                # Update status to FAILED
                self.metadata_store.update_file_status(file_id, FileStatus.FAILED, **kwargs)
                return False

            # If validation passes and status was FAILED, update to STORED
            if metadata.status == FileStatus.FAILED:
                self.metadata_store.update_file_status(file_id, FileStatus.STORED, **kwargs)

            logger.info(f"File upload validation successful: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate file upload {file_id}: {e}")
            return False

    def _cleanup_failed_file_upload(self, file_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed file upload"""
        try:
            cleanup_success = True

            # Get metadata to find blob key
            metadata = self.metadata_store.get_file_metadata(file_id, **kwargs)
            if metadata:
                # Try to delete blob
                try:
                    if self.blob_store.exists(metadata.blob_key, **kwargs):
                        blob_deleted = self.blob_store.delete(metadata.blob_key, **kwargs)
                        if not blob_deleted:
                            logger.warning(f"Failed to delete blob during cleanup: {metadata.blob_key}")
                            cleanup_success = False
                        else:
                            logger.info(f"Deleted blob during cleanup: {metadata.blob_key}")
                except Exception as blob_error:
                    logger.error(f"Error deleting blob during cleanup: {blob_error}")
                    cleanup_success = False

            # Delete metadata
            try:
                metadata_deleted = self.metadata_store.delete_file_metadata(file_id, **kwargs)
                if not metadata_deleted:
                    logger.warning(f"Failed to delete metadata during cleanup: {file_id}")
                    cleanup_success = False
                else:
                    logger.info(f"Deleted metadata during cleanup: {file_id}")
            except Exception as metadata_error:
                logger.error(f"Error deleting metadata during cleanup: {metadata_error}")
                cleanup_success = False

            return cleanup_success

        except Exception as e:
            logger.error(f"Failed to cleanup failed file upload {file_id}: {e}")
            return False

    def upload_file(
        self,
        filename: str,
        file_data: bytes,
        owner_id: uuid.UUID,
        content_type: Optional[str] = None,
        validate_after_store: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Upload a single file with validation and coordination.

        Args:
            filename: Original filename
            file_data: Binary file data
            owner_id: UUID of the user who owns this file
            content_type: MIME type of the file
            validate_after_store: Whether to validate after storing
            **kwargs: Additional arguments

        Returns:
            str: File ID of the uploaded file

        Raises:
            FileValidationError: If file validation fails
            StorageOperationError: If storage operation fails
        """
        try:
            # Validate parameters
            self._validate_file_upload(filename, file_data)
            logger.info(f"Validated file upload request: {filename}")

            # Generate IDs and keys
            file_id = self._generate_file_id()
            blob_key = self._generate_blob_key(file_id, filename)

            # Calculate file properties
            file_size = len(file_data)
            content_type = content_type or "application/octet-stream"

            # Create metadata object with STORED status
            now = datetime.now(tz=datetime.now().astimezone().tzinfo)
            metadata = FileMetadata(
                file_id=file_id,
                owner_id=owner_id,
                blob_key=blob_key,
                filename=filename,
                status=FileStatus.STORED,
                file_size=file_size,
                content_type=content_type,
                created_at=now,
                updated_at=now
            )

            # Store metadata first through metadata store
            logger.info(f"Storing metadata for file: {filename} (file_id: {file_id})")
            stored_metadata_id = self.metadata_store.store_file_metadata(metadata, **kwargs)
            assert stored_metadata_id == file_id

            try:
                # Store blob data through blob store
                logger.info(f"Storing blob data for file: {filename} (key: {blob_key})")
                stored_blob_key, was_overwritten = self.blob_store.store(
                    blob_key,
                    file_data,
                    content_type=content_type,
                    **kwargs
                )

                # Update metadata with final blob key
                self.metadata_store.update_file_metadata(
                    file_id,
                    {
                        'blob_key': stored_blob_key,  # Use actual stored key (may be versioned)
                        'status': FileStatus.STORED,
                        'updated_at': datetime.now(tz=datetime.now().astimezone().tzinfo)
                    },
                    **kwargs
                )

                if was_overwritten:
                    logger.warning(f"Blob was overwritten during storage: {stored_blob_key}")

                logger.info(f"Successfully stored file: {filename} (file_id: {file_id}, blob_key: {stored_blob_key})")

            except Exception as blob_error:
                # Blob storage failed, update metadata status to FAILED
                logger.error(f"Blob storage failed for {filename}: {blob_error}")
                self.metadata_store.update_file_status(file_id, FileStatus.FAILED, **kwargs)
                raise StorageOperationError(f"Blob storage failed: {str(blob_error)}")

            # Validate after storage if requested
            if validate_after_store:
                validation_passed = self._validate_stored_file(file_id, **kwargs)

                if not validation_passed:
                    # Validation failed, cleanup
                    logger.error(f"File validation failed: {filename}")
                    cleanup_success = self._cleanup_failed_file_upload(file_id, **kwargs)
                    if cleanup_success:
                        logger.info(f"Cleaned up failed upload: {file_id}")
                    else:
                        logger.warning(f"Failed to cleanup after validation failure: {file_id}")

                    raise StorageOperationError("File validation failed after storage")

                logger.info(f"File validation passed: {filename}")

            return file_id

        except FileValidationError:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            logger.error(f"Storage error for {filename}: {e}")
            raise StorageOperationError(f"Storage error: {str(e)}")

    def get_file_metadata(self, file_id: str, **kwargs: Any) -> Optional['FileMetadata']:
        """Retrieve file metadata by file ID"""
        try:
            return self.metadata_store.get_file_metadata(file_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get file metadata for {file_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve file metadata: {e}")

    def get_file_content(self, file_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve file content by file ID"""
        try:
            # Get metadata to find blob key
            metadata = self.metadata_store.get_file_metadata(file_id, **kwargs)
            if not metadata:
                logger.warning(f"File metadata not found for file_id: {file_id}")
                return None

            # Retrieve blob content
            try:
                content = self.blob_store.retrieve(metadata.blob_key, **kwargs)
                logger.debug(f"Retrieved file content for file_id: {file_id}")
                return content
            except KeyError:
                logger.error(f"Blob not found for file_id: {file_id}, blob_key: {metadata.blob_key}")
                return None

        except Exception as e:
            logger.error(f"Failed to get file content for {file_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve file content: {e}")

    def update_file_metadata(self, file_id: str, **kwargs: Any) -> bool:
        """Update file metadata by file ID"""
        try:
            result = self.metadata_store.update_file_metadata(file_id, kwargs, **kwargs)
            if result:
                logger.info(f"Updated file metadata: {file_id}")
            else:
                logger.warning(f"Failed to update file metadata: {file_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update file metadata for {file_id}: {e}")
            raise StorageOperationError(f"Failed to update file metadata: {e}")

    def delete_file(self, file_id: str, **kwargs: Any) -> bool:
        """Delete file and cleanup associated data"""
        try:
            return self._cleanup_failed_file_upload(file_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise StorageOperationError(f"Failed to delete file: {e}")

    def list_files_by_owner(
        self,
        owner_id: uuid.UUID,
        status: Optional['FileStatus'] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any
    ) -> List['FileMetadata']:
        """
        List all files for a specific owner with optional filtering.

        Args:
            owner_id: UUID of the file owner
            status: Optional file status filter
            limit: Maximum number of records to return
            offset: Number of records to skip
            **kwargs: Additional arguments

        Returns:
            List of FileMetadata objects

        Raises:
            StorageOperationError: If listing operation fails
        """
        try:
            files = self.metadata_store.list_file_metadata(
                owner_id=owner_id,
                status=status,
                limit=limit,
                offset=offset,
                **kwargs
            )
            logger.debug(f"Retrieved {len(files)} files for owner {owner_id}")
            return files
        except Exception as e:
            logger.error(f"Failed to list files for owner {owner_id}: {e}")
            raise StorageOperationError(f"Failed to list files: {e}")

    def count_files(
        self,
        owner_id: uuid.UUID | None = None,
        status: FileStatus | None = None
    ) -> int:
        """
        Count all files for a specific owner with optional filtering.

        Args:
            owner_id: UUID of the file owner
            status: Optional file status filter

        Returns:
            Total count of files for the owner

        Raises:
            StorageOperationError: If counting operation fails
        """
        try:
            count = self.metadata_store.count_file_metadata(owner_id, status)
            return count
        except Exception as e:
            logger.error(f"Failed to count files for owner {owner_id}: {e}")
            raise StorageOperationError(f"Failed to count files: {e}")

    def list_accessible_files(
        self,
        user_id: uuid.UUID,
        status: Optional['FileStatus'] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List['FileMetadata']:
        """
        List all files accessible to a user (owned files + files with permissions).

        Args:
            user_id: UUID of the user to check access for
            status: Optional file status filter
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of FileMetadata objects accessible to the user

        Raises:
            StorageOperationError: If listing operation fails
        """
        try:
            files = self.metadata_store.list_accessible_files(
                user_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            logger.debug(f"Retrieved {len(files)} accessible files for user {user_id}")
            return files
        except Exception as e:
            logger.exception(f"Failed to list accessible files for user {user_id}")
            raise StorageOperationError(f"Failed to list accessible files: {e}")

    def count_accessible_files(
        self,
        user_id: uuid.UUID,
        status: Optional['FileStatus'] = None
    ) -> int:
        """
        Count all files accessible to a user (owned files + files with permissions).

        Args:
            user_id: UUID of the user to check access for
            status: Optional file status filter
            **kwargs: Additional arguments

        Returns:
            Total count of files accessible to the user

        Raises:
            StorageOperationError: If counting operation fails
        """
        try:
            count = self.metadata_store.count_accessible_files(
                user_id=user_id,
                status=status
            )
            logger.debug(f"Counted {count} accessible files for user {user_id}")
            return count
        except Exception as e:
            logger.exception(f"Failed to count accessible files for user {user_id}")
            raise StorageOperationError(f"Failed to count accessible files: {e}")
