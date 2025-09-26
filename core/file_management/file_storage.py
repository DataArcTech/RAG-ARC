from typing import (
    Any,
    Optional,
    List,
    Dict,
)
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import uuid

from encapsulation.data_model.orm_models import FileMetadata, ParsedContentMetadata, ChunkMetadata
from encapsulation.data_model.orm_models import FileStatus, ParsedContentStatus, ChunkIndexStatus

from framework.module import AbstractModule

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

    Provides high-level file upload, validation, and parsed content storage
    operations with coordination between blob storage and metadata storage.

    Key features:
    - File validation and metadata verification
    - Parsed content storage linked to source files
    - Chunk storage linked to parsed content
    - Automatic cleanup on validation failures
    - Upload session tracking
    - Comprehensive error handling and reporting

    Architecture:
        Application Layer -> FileStorage (Core) -> Blob Storage + Metadata Storage

    Dependencies:
        blob_store: FileDB implementation (e.g., LocalDB, MinIODB)
        metadata_store: RelationalDB implementation (e.g., PostgreSQLDB)
    """

    def __init__(self, config):
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

    def _generate_parsed_content_id(self) -> str:
        """Generate unique parsed content ID"""
        return str(uuid.uuid4())

    def _generate_parsed_blob_key(self, parsed_content_id: str, source_file_id: str, parser_type: str) -> str:
        """Generate blob storage key for parsed content"""
        # Create hierarchical key: parsed/{first-2-chars-of-source-id}/{source-file-id}/{parsed-content-id}.{parser-type}
        prefix = source_file_id[:2]
        return f"parsed/{prefix}/{source_file_id}/{parsed_content_id}.{parser_type}"

    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID"""
        return str(uuid.uuid4())

    def _generate_chunk_blob_key(self, chunk_id: str, source_parsed_content_id: str, chunker_type: str) -> str:
        """Generate blob storage key for chunk"""
        # Create hierarchical key: chunks/{first-2-chars-of-source-id}/{source-parsed-content-id}/{chunk-id}.json
        prefix = source_parsed_content_id[:2]
        return f"chunks/{prefix}/{source_parsed_content_id}/{chunk_id}.json"
    
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

    def _validate_stored_parsed_content(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Validate that parsed content was stored successfully"""
        try:
            # Get parsed content metadata
            metadata = self.metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
            if not metadata:
                logger.warning(f"Parsed content metadata not found for validation: {parsed_content_id}")
                return False

            # Check if blob exists
            blob_exists = self.blob_store.exists(metadata.blob_key, **kwargs)
            if not blob_exists:
                logger.error(f"Parsed content blob validation failed - blob not found: {metadata.blob_key}")
                self.metadata_store.update_parsed_content_status(parsed_content_id, ParsedContentStatus.FAILED, **kwargs)
                return False

            # If validation passes and status was FAILED, update to STORED
            if metadata.status == ParsedContentStatus.FAILED:
                self.metadata_store.update_parsed_content_status(parsed_content_id, ParsedContentStatus.STORED, **kwargs)

            logger.info(f"Parsed content upload validation successful: {parsed_content_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate parsed content upload {parsed_content_id}: {e}")
            return False

    def _cleanup_failed_parsed_content_upload(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed parsed content upload"""
        try:
            cleanup_success = True

            # Get metadata to find blob key
            metadata = self.metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
            if metadata:
                # Try to delete blob
                try:
                    if self.blob_store.exists(metadata.blob_key, **kwargs):
                        blob_deleted = self.blob_store.delete(metadata.blob_key, **kwargs)
                        if not blob_deleted:
                            logger.warning(f"Failed to delete parsed content blob during cleanup: {metadata.blob_key}")
                            cleanup_success = False
                        else:
                            logger.info(f"Deleted parsed content blob during cleanup: {metadata.blob_key}")
                except Exception as blob_error:
                    logger.error(f"Error deleting parsed content blob during cleanup: {blob_error}")
                    cleanup_success = False

            # Delete metadata
            try:
                metadata_deleted = self.metadata_store.delete_parsed_content_metadata(parsed_content_id, **kwargs)
                if not metadata_deleted:
                    logger.warning(f"Failed to delete parsed content metadata during cleanup: {parsed_content_id}")
                    cleanup_success = False
                else:
                    logger.info(f"Deleted parsed content metadata during cleanup: {parsed_content_id}")
            except Exception as metadata_error:
                logger.error(f"Error deleting parsed content metadata during cleanup: {metadata_error}")
                cleanup_success = False

            return cleanup_success

        except Exception as e:
            logger.error(f"Failed to cleanup failed parsed content upload {parsed_content_id}: {e}")
            return False

    def _cleanup_failed_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed chunk upload"""
        try:
            cleanup_success = True

            # Get metadata to find blob key
            metadata = self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if metadata:
                # Try to delete blob
                try:
                    if self.blob_store.exists(metadata.blob_key, **kwargs):
                        blob_deleted = self.blob_store.delete(metadata.blob_key, **kwargs)
                        if not blob_deleted:
                            logger.warning(f"Failed to delete chunk blob during cleanup: {metadata.blob_key}")
                            cleanup_success = False
                        else:
                            logger.info(f"Deleted chunk blob during cleanup: {metadata.blob_key}")
                except Exception as blob_error:
                    logger.error(f"Error deleting chunk blob during cleanup: {blob_error}")
                    cleanup_success = False

            # Delete metadata
            try:
                metadata_deleted = self.metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
                if not metadata_deleted:
                    logger.warning(f"Failed to delete chunk metadata during cleanup: {chunk_id}")
                    cleanup_success = False
                else:
                    logger.info(f"Deleted chunk metadata during cleanup: {chunk_id}")
            except Exception as metadata_error:
                logger.error(f"Error deleting chunk metadata during cleanup: {metadata_error}")
                cleanup_success = False

            return cleanup_success

        except Exception as e:
            logger.error(f"Failed to cleanup failed chunk upload {chunk_id}: {e}")
            return False

    def _validate_stored_chunk(self, chunk_id: str, **kwargs: Any) -> bool:
        """Validate that chunk was stored successfully"""
        try:
            # Get chunk metadata
            metadata = self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if not metadata:
                logger.warning(f"Chunk metadata not found for validation: {chunk_id}")
                return False

            # Check if blob exists
            blob_exists = self.blob_store.exists(metadata.blob_key, **kwargs)
            if not blob_exists:
                logger.error(f"Chunk blob validation failed - blob not found: {metadata.blob_key}")
                return False

            logger.info(f"Chunk upload validation successful: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate chunk upload {chunk_id}: {e}")
            return False
    
    def upload_file(
        self,
        filename: str,
        file_data: bytes,
        content_type: Optional[str] = None,
        validate_after_store: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Upload a single file with validation and coordination.

        Args:
            filename: Original filename
            file_data: Binary file data
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
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            metadata = FileMetadata(
                file_id=file_id,
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
                        'updated_at': datetime.now(tz=ZoneInfo("Asia/Shanghai"))
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

    def store_parsed_content(
        self,
        source_file_id: str,
        parser_type: str,
        parsed_data: bytes,
        content_type: str = "text/markdown",
        validate_after_store: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Store parsed content linked to source file with coordination.

        Args:
            source_file_id: ID of the original file that was parsed
            parser_type: Type of parser used (e.g., "dots_ocr", "pypdf")
            parsed_data: Binary parsed content data
            content_type: MIME type of parsed content (default: "text/markdown")
            validate_after_store: Whether to validate after storing
            **kwargs: Additional arguments

        Returns:
            str: Parsed content ID of the stored parsed content

        Raises:
            FileValidationError: If validation fails
            StorageOperationError: If storage operation fails
        """
        # Validate parameters
        if not parsed_data:
            raise FileValidationError("parsed_data must be provided")

        try:
            # Verify source file exists
            source_metadata = self.metadata_store.get_file_metadata(source_file_id, **kwargs)
            if not source_metadata:
                raise ValueError(f"Source file {source_file_id} not found")

            # Generate IDs and keys
            parsed_content_id = self._generate_parsed_content_id()
            blob_key = self._generate_parsed_blob_key(parsed_content_id, source_file_id, parser_type)

            # Create parsed content metadata object
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            parsed_metadata = ParsedContentMetadata(
                parsed_content_id=parsed_content_id,
                source_file_id=source_file_id,
                blob_key=blob_key,
                parser_type=parser_type,
                status=ParsedContentStatus.STORED,
                created_at=now,
                updated_at=now,
                content_type=content_type
            )

            # Store parsed metadata first
            logger.info(f"Storing parsed content metadata: {parsed_content_id} (source: {source_file_id})")
            stored_metadata_id = self.metadata_store.store_parsed_content_metadata(parsed_metadata, **kwargs)
            assert stored_metadata_id == parsed_content_id

            try:
                # Store parsed content blob
                logger.info(f"Storing parsed content blob: {blob_key}")
                stored_blob_key, was_overwritten = self.blob_store.store(
                    blob_key,
                    parsed_data,
                    content_type=content_type,
                    **kwargs
                )

                # Update metadata with final blob key
                self.metadata_store.update_parsed_content_metadata(
                    parsed_content_id,
                    {
                        'blob_key': stored_blob_key,
                        'status': ParsedContentStatus.STORED,
                        'updated_at': datetime.now(tz=ZoneInfo("Asia/Shanghai"))
                    },
                    **kwargs
                )

                if was_overwritten:
                    logger.warning(f"Parsed content blob was overwritten: {stored_blob_key}")

                logger.info(f"Successfully stored parsed content: {parsed_content_id} (blob_key: {stored_blob_key})")

            except Exception as blob_error:
                # Blob storage failed, update metadata status to FAILED
                logger.error(f"Parsed content blob storage failed: {blob_error}")
                self.metadata_store.update_parsed_content_status(parsed_content_id, ParsedContentStatus.FAILED, **kwargs)
                raise StorageOperationError(f"Parsed content blob storage failed: {str(blob_error)}")

            # Validate after storage if requested
            if validate_after_store:
                validation_passed = self._validate_stored_parsed_content(parsed_content_id, **kwargs)

                if not validation_passed:
                    # Validation failed, cleanup
                    logger.error(f"Parsed content validation failed: {parsed_content_id}")
                    cleanup_success = self._cleanup_failed_parsed_content_upload(parsed_content_id, **kwargs)
                    if cleanup_success:
                        logger.info(f"Cleaned up failed parsed content: {parsed_content_id}")
                    else:
                        logger.warning(f"Failed to cleanup after validation failure: {parsed_content_id}")

                    raise StorageOperationError("Parsed content validation failed after storage")

                logger.info(f"Parsed content validation passed: {parsed_content_id}")

            return parsed_content_id

        except FileValidationError:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            logger.error(f"Parsed content storage error: {e}")
            raise StorageOperationError(f"Storage error: {str(e)}")
    

    def store_chunk(
        self,
        source_parsed_content_id: str,
        chunker_type: str,
        chunk_data: bytes,
        validate_after_store: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Store a single chunk linked to parsed content with coordination.

        Args:
            source_parsed_content_id: ID of the parsed content that was chunked
            chunker_type: Type of chunker used (e.g., "semantic_chunker", "token_chunker")
            chunk_data: Binary chunk data (JSON format)
            validate_after_store: Whether to validate after storing
            **kwargs: Additional arguments

        Returns:
            str: Chunk ID of the stored chunk

        Raises:
            FileValidationError: If validation fails
            StorageOperationError: If storage operation fails
        """
        # Validate parameters
        if not chunk_data:
            raise FileValidationError("chunk_data must be provided")

        try:
            # Verify source parsed content exists
            source_metadata = self.metadata_store.get_parsed_content_metadata(source_parsed_content_id, **kwargs)
            if not source_metadata:
                raise ValueError(f"Source parsed content {source_parsed_content_id} not found")

            # Generate IDs and keys
            chunk_id = self._generate_chunk_id()
            blob_key = self._generate_chunk_blob_key(chunk_id, source_parsed_content_id, chunker_type)

            # Create chunk metadata object
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source_parsed_content_id=source_parsed_content_id,
                blob_key=blob_key,
                chunker_type=chunker_type,
                index_status=ChunkIndexStatus.STORED,
                created_at=now
            )

            # Store chunk metadata first
            logger.info(f"Storing chunk metadata: {chunk_id} (source: {source_parsed_content_id})")
            stored_metadata_id = self.metadata_store.store_chunk_metadata(chunk_metadata, **kwargs)
            assert stored_metadata_id == chunk_id

            try:
                # Store chunk blob
                logger.info(f"Storing chunk blob: {blob_key}")
                stored_blob_key, was_overwritten = self.blob_store.store(
                    blob_key,
                    chunk_data,
                    content_type="application/json",
                    **kwargs
                )

                # Update metadata with final blob key
                self.metadata_store.update_chunk_metadata(
                    chunk_id,
                    {
                        'blob_key': stored_blob_key
                    },
                    **kwargs
                )

                if was_overwritten:
                    logger.warning(f"Chunk blob was overwritten: {stored_blob_key}")

                logger.info(f"Successfully stored chunk: {chunk_id} (blob_key: {stored_blob_key})")

            except Exception as blob_error:
                # Blob storage failed, delete metadata
                logger.error(f"Chunk blob storage failed: {blob_error}")
                self.metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
                raise StorageOperationError(f"Chunk blob storage failed: {str(blob_error)}")

            # Validate after storage if requested
            if validate_after_store:
                validation_passed = self._validate_stored_chunk(chunk_id, **kwargs)

                if not validation_passed:
                    # Validation failed, cleanup
                    logger.error(f"Chunk validation failed: {chunk_id}")
                    cleanup_success = self._cleanup_failed_chunk_upload(chunk_id, **kwargs)
                    if cleanup_success:
                        logger.info(f"Cleaned up failed chunk: {chunk_id}")
                    else:
                        logger.warning(f"Failed to cleanup after validation failure: {chunk_id}")

                    raise StorageOperationError("Chunk validation failed after storage")

                logger.info(f"Chunk validation passed: {chunk_id}")

            return chunk_id

        except FileValidationError:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            logger.error(f"Chunk storage error: {e}")
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
    
    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional['ParsedContentMetadata']:
        """Retrieve parsed content metadata by ID"""
        try:
            return self.metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get parsed content metadata for {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve parsed content metadata: {e}")
    
    def get_parsed_content(self, parsed_content_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve parsed content by ID"""
        try:
            # Get metadata to find blob key
            metadata = self.metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
            if not metadata:
                logger.warning(f"Parsed content metadata not found for id: {parsed_content_id}")
                return None

            # Retrieve blob content
            try:
                content = self.blob_store.retrieve(metadata.blob_key, **kwargs)
                logger.debug(f"Retrieved parsed content for id: {parsed_content_id}")
                return content
            except KeyError:
                logger.error(f"Parsed content blob not found for id: {parsed_content_id}, blob_key: {metadata.blob_key}")
                return None

        except Exception as e:
            logger.error(f"Failed to get parsed content for {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve parsed content: {e}")

    def get_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional['ChunkMetadata']:
        """Retrieve chunk metadata by ID"""
        try:
            return self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get chunk metadata for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve chunk metadata: {e}")

    def get_chunk_content(self, chunk_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve chunk content by ID"""
        try:
            # Get metadata to find blob key
            metadata = self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if not metadata:
                logger.warning(f"Chunk metadata not found for id: {chunk_id}")
                return None

            # Retrieve blob content
            try:
                content = self.blob_store.retrieve(metadata.blob_key, **kwargs)
                logger.debug(f"Retrieved chunk content for id: {chunk_id}")
                return content
            except KeyError:
                logger.error(f"Chunk blob not found for id: {chunk_id}, blob_key: {metadata.blob_key}")
                return None

        except Exception as e:
            logger.error(f"Failed to get chunk content for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve chunk content: {e}")

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

    def update_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Update parsed content metadata by ID"""
        try:
            result = self.metadata_store.update_parsed_content_metadata(parsed_content_id, kwargs, **kwargs)
            if result:
                logger.info(f"Updated parsed content metadata: {parsed_content_id}")
            else:
                logger.warning(f"Failed to update parsed content metadata: {parsed_content_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update parsed content metadata for {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to update parsed content metadata: {e}")

    def update_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> bool:
        """Update chunk metadata by ID"""
        try:
            result = self.metadata_store.update_chunk_metadata(chunk_id, kwargs, **kwargs)
            if result:
                logger.info(f"Updated chunk metadata: {chunk_id}")
            else:
                logger.warning(f"Failed to update chunk metadata: {chunk_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update chunk metadata for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to update chunk metadata: {e}")

    def delete_file(self, file_id: str, **kwargs: Any) -> bool:
        """Delete file and cleanup associated data"""
        try:
            return self._cleanup_failed_file_upload(file_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise StorageOperationError(f"Failed to delete file: {e}")
    
    def delete_parsed_content(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Delete parsed content and cleanup associated data"""
        try:
            return self._cleanup_failed_parsed_content_upload(parsed_content_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete parsed content {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to delete parsed content: {e}")

    def delete_chunk(self, chunk_id: str, **kwargs: Any) -> bool:
        """Delete chunk and cleanup associated data"""
        try:
            return self._cleanup_failed_chunk_upload(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to delete chunk: {e}")