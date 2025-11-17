from typing import (
    Any,
    Optional,
    TYPE_CHECKING,
)
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import uuid

from encapsulation.data_model.orm_models import ParsedContentMetadata
from encapsulation.data_model.orm_models import ParsedContentStatus

from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.core.file_management.storage.parsed_content_storage import ParsedContentStorage

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails"""
    pass


class StorageOperationError(Exception):
    """Raised when storage operation fails"""
    pass


class ParsedContentStorage(AbstractModule):
    """
    Core parsed content storage interface for RAG system.

    Provides high-level parsed content storage operations with coordination
    between blob storage and metadata storage.

    Key features:
    - Parsed content storage linked to source files
    - Automatic cleanup on validation failures
    - Comprehensive error handling and reporting

    Architecture:
        Application Layer -> ParsedContentStorage (Core) -> Blob Storage + Metadata Storage

    Dependencies:
        blob_store: FileDB implementation (e.g., LocalDB, MinIODB)
        metadata_store: RelationalDB implementation (e.g., PostgreSQLDB)
    """

    def __init__(self, config):
        """Initialize ParsedContentStorage with eager blob and metadata store creation"""
        super().__init__(config)
        # Build stores directly (no intermediate data_store layer)
        self.blob_store = config.file_db_config.build()
        self.metadata_store = config.relational_db_config.build()

    def _generate_parsed_content_id(self) -> str:
        """Generate unique parsed content ID"""
        return str(uuid.uuid4())

    def _generate_parsed_blob_key(self, parsed_content_id: str, source_file_id: str, parser_type: str) -> str:
        """Generate blob storage key for parsed content"""
        # Create hierarchical key: parsed/{first-2-chars-of-source-id}/{source-file-id}/{parsed-content-id}.{parser-type}
        prefix = source_file_id[:2]
        return f"parsed/{prefix}/{source_file_id}/{parsed_content_id}.{parser_type}"

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
            now = datetime.now(tz=datetime.now().astimezone().tzinfo)
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
                        'updated_at': datetime.now(tz=datetime.now().astimezone().tzinfo)
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

    def delete_parsed_content(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Delete parsed content and cleanup associated data"""
        try:
            return self._cleanup_failed_parsed_content_upload(parsed_content_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete parsed content {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to delete parsed content: {e}")
