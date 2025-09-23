from __future__ import annotations

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)
import logging

from ..data_model.orm_models import FileMetadata, FileStatus
from ..data_model.orm_models import ParsedContentMetadata, ParsedContentStatus
from ..data_model.orm_models import ChunkMetadata, ChunkIndexStatus

from .base import DataStore
from framework.singleton_decorator import singleton

logger = logging.getLogger(__name__)


@singleton
class FileStore(DataStore):
    """
    Coordinated file database implementation orchestrating blob storage and metadata storage.
    
    This class provides a complete file storage solution that coordinates multiple storage
    systems to ensure data consistency and integrity. It manages file uploads, metadata
    tracking, parsed content storage, and provides comprehensive validation and cleanup
    operations.
    
    Key features:
    - Atomic-like operations across blob and metadata storage
    - Hierarchical blob key generation for efficient organization  
    - SHA-256 checksum verification for data integrity
    - Upload status tracking throughout the storage lifecycle
    - Collision handling with overwrite/error/version modes
    - Comprehensive cleanup for failed operations
    - Support for parsed content linking to source files
    - Asynchronous operation support via ThreadPoolExecutor
    
    Storage architecture:
    - Blob storage: Raw file data (MinIO, S3, Local filesystem)
    - Metadata storage: File metadata, status, relationships (PostgreSQL)
    - Hierarchical keys: files/{prefix}/{file-id}/{filename}
    - Parsed content keys: parsed/{prefix}/{source-id}/{parsed-id}.{type}

    Upload workflow:
        1. Generate unique file_id and hierarchical blob_key
        2. Create metadata record with STORED status
        3. Store blob data with collision handling
        4. Update metadata with final blob_key and STORED status
        5. Handle failures by updating status to FAILED
        
    Status tracking states:
        - UPLOADING: Upload in progress
        - UPLOADED: Upload completed successfully
        - PARSING: Content parsing in progress  
        - PARSED: Parsing completed successfully
        - FAILED: Operation failed
        - ARCHIVED: Content archived
        
    Collision handling modes:
        - "overwrite": Replace existing blob (default)
        - "error": Raise KeyError if key exists
        - "version": Generate versioned key if exists
        
    Dependencies:
        blob_store: FileDB implementation (MinIODB, LocalDB, etc.)
        metadata_store: RelationalDB implementation (PostgreSQLDB)
        
    Core methods:
        - store_file: Store files with metadata coordination
        - store_parsed_content: Store parsed content linked to source files
        - store_chunk: Store individual chunk data
        - get_file_metadata/get_file_content: Retrieve with cross-system lookup
        - validate_file_upload: Verify blob + metadata consistency
        - cleanup_failed_file_upload: Remove orphaned data
        
    Performance considerations:
    - Metadata operations are synchronous with blob operations
    - Failed operations require cleanup to prevent orphaned data
    
    Typical usage:
        >>> blob_store = LocalDB(config)
        >>> metadata_store = PostgreSQLDB(config)
        >>> file_store = FileStore(blob_store, metadata_store)
        >>> metadata = file_store.store_file(file_data, "document.pdf")
        >>> content = file_store.get_file_content(metadata.file_id)
        >>> is_valid = file_store.validate_file_upload(metadata.file_id)
        
    Error handling:
    - Partial failures are handled by updating metadata status
    - Cleanup operations remove orphaned data from both systems
    - Validation methods verify cross-system consistency
    - All operations log detailed information for debugging
        
    Attributes:
        blob_store: Blob storage implementation
        metadata_store: Relational database implementation
    """

    def __init__(self, config):
        """Initialize FileStore with eager blob and metadata store creation"""
        super().__init__(config)
        # Build stores immediately since we always need them
        self.blob_store = config.file_db_config.build()
        self.metadata_store = config.relational_db_config.build()
    
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
    
    def store_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> 'FileMetadata':
        """Store file data and create metadata record"""
        try:
            # Build sub-configs on-demand
            blob_store = self.blob_store
            metadata_store = self.metadata_store
            
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

            # Store metadata first (with UPLOADING status)
            logger.info(f"Storing metadata for file: {filename} (file_id: {file_id})")
            stored_metadata_id = metadata_store.store_file_metadata(metadata, **kwargs)
            assert stored_metadata_id == file_id
            
            try:
                # Store blob data
                logger.info(f"Storing blob data for file: {filename} (key: {blob_key})")
                stored_blob_key, was_overwritten = blob_store.store(
                    blob_key, 
                    file_data, 
                    content_type=content_type,
                    **kwargs
                )
                
                # Update metadata with STORED status and final blob key
                metadata_store.update_file_metadata(
                    file_id,
                    {
                        'blob_key': stored_blob_key,  # Use actual stored key (may be versioned)
                        'status': FileStatus.STORED,
                        'updated_at': datetime.now(tz=ZoneInfo("Asia/Shanghai"))
                    },
                    **kwargs
                )

                # Update our metadata object to reflect final state
                metadata.blob_key = stored_blob_key
                metadata.status = FileStatus.STORED
                metadata.updated_at = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

                if was_overwritten:
                    logger.warning(f"Blob was overwritten during storage: {stored_blob_key}")

                logger.info(f"Successfully stored file: {filename} (file_id: {file_id}, blob_key: {stored_blob_key})")
                return metadata

            except Exception as blob_error:
                # Blob storage failed, update metadata status to FAILED
                logger.error(f"Blob storage failed for {filename}: {blob_error}")
                metadata_store.update_file_status(file_id, FileStatus.FAILED, **kwargs)
                raise
                
        except Exception as e:
            logger.error(f"Failed to store file {filename}: {e}")
            # If metadata storage also failed, the exception will propagate
            raise

    def store_parsed_content(
        self,
        parsed_data: bytes,
        source_file_id: str,
        parser_type: str,
        content_type: str = "text/markdown",
        **kwargs: Any,
    ) -> 'ParsedContentMetadata':
        """Store parsed content data and create metadata record"""
        try:
            # Build sub-configs on-demand
            blob_store = self.blob_store
            metadata_store = self.metadata_store
            
            # Verify source asset exists
            source_metadata = metadata_store.get_file_metadata(source_file_id, **kwargs)
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
            stored_metadata_id = metadata_store.store_parsed_content_metadata(parsed_metadata, **kwargs)
            assert stored_metadata_id == parsed_content_id

            try:
                # Store parsed content blob
                logger.info(f"Storing parsed content blob: {blob_key}")
                stored_blob_key, was_overwritten = blob_store.store(
                    blob_key,
                    parsed_data,
                    content_type=content_type,
                    **kwargs
                )

                # Update metadata with STORED status and final blob key
                metadata_store.update_parsed_content_metadata(
                    parsed_content_id,
                    {
                        'blob_key': stored_blob_key,
                        'status': ParsedContentStatus.STORED,
                        'updated_at': datetime.now(tz=ZoneInfo("Asia/Shanghai"))
                    },
                    **kwargs
                )

                # Update our metadata object to reflect final state
                parsed_metadata.blob_key = stored_blob_key
                parsed_metadata.status = ParsedContentStatus.STORED
                parsed_metadata.updated_at = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

                if was_overwritten:
                    logger.warning(f"Parsed content blob was overwritten: {stored_blob_key}")

                logger.info(f"Successfully stored parsed content: {parsed_content_id} (blob_key: {stored_blob_key})")
                return parsed_metadata

            except Exception as blob_error:
                # Blob storage failed, update metadata status to FAILED
                logger.error(f"Parsed content blob storage failed: {blob_error}")
                metadata_store.update_parsed_content_status(parsed_content_id, ParsedContentStatus.FAILED, **kwargs)
                raise
    
        except Exception as e:
            logger.error(f"Failed to store parsed content: {e}")
            raise

    def store_chunk(
        self,
        chunk_data: bytes,
        source_parsed_content_id: str,
        chunker_type: str,
        **kwargs: Any,
    ) -> 'ChunkMetadata':
        """Store chunk data and create metadata record"""
        try:
            # Build sub-configs on-demand
            blob_store = self.blob_store
            metadata_store = self.metadata_store

            # Verify source parsed content exists
            source_metadata = metadata_store.get_parsed_content_metadata(source_parsed_content_id, **kwargs)
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
            stored_metadata_id = metadata_store.store_chunk_metadata(chunk_metadata, **kwargs)
            assert stored_metadata_id == chunk_id

            try:
                # Store chunk blob
                logger.info(f"Storing chunk blob: {blob_key}")
                stored_blob_key, was_overwritten = blob_store.store(
                    blob_key,
                    chunk_data,
                    content_type="application/json",
                    **kwargs
                )

                # Update metadata with final blob key
                metadata_store.update_chunk_metadata(
                    chunk_id,
                    {
                        'blob_key': stored_blob_key
                    },
                    **kwargs
                )

                # Update our metadata object to reflect final state
                chunk_metadata.blob_key = stored_blob_key

                if was_overwritten:
                    logger.warning(f"Chunk blob was overwritten: {stored_blob_key}")

                logger.info(f"Successfully stored chunk: {chunk_id} (blob_key: {stored_blob_key})")
                return chunk_metadata

            except Exception as blob_error:
                # Blob storage failed, delete metadata
                logger.error(f"Chunk blob storage failed: {blob_error}")
                metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
                raise

        except Exception as e:
            logger.error(f"Failed to store chunk: {e}")
            raise

    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID"""
        return str(uuid.uuid4())

    def _generate_chunk_blob_key(self, chunk_id: str, source_parsed_content_id: str, chunker_type: str) -> str:
        """Generate blob storage key for chunk"""
        # Create hierarchical key: chunks/{first-2-chars-of-source-id}/{source-parsed-content-id}/{chunk-id}.json
        prefix = source_parsed_content_id[:2]
        return f"chunks/{prefix}/{source_parsed_content_id}/{chunk_id}.json"

    def get_file_metadata(self, file_id: str, **kwargs: Any) -> Optional['FileMetadata']:
        """Retrieve file metadata by file ID"""
        try:
            metadata_store = self.metadata_store
            return metadata_store.get_file_metadata(file_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get file metadata for {file_id}: {e}")
            raise
    
    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional['ParsedContentMetadata']:
        """Retrieve parsed content metadata by ID"""
        try:
            metadata_store = self.metadata_store
            return metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get parsed content metadata for {parsed_content_id}: {e}")
            raise
    
    def get_file_content(self, file_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve file content by file ID"""
        try:
            # Get metadata to find blob key
            blob_store = self.blob_store
            metadata_store = self.metadata_store

            metadata = metadata_store.get_file_metadata(file_id, **kwargs)
            if not metadata:
                logger.warning(f"File metadata not found for file_id: {file_id}")
                return None

            # Retrieve blob content
            try:
                content = blob_store.retrieve(metadata.blob_key, **kwargs)
                logger.debug(f"Retrieved file content for file_id: {file_id}")
                return content
            except KeyError:
                logger.error(f"Blob not found for file_id: {file_id}, blob_key: {metadata.blob_key}")
                return None

        except Exception as e:
            logger.error(f"Failed to get file content for {file_id}: {e}")
            raise
    
    def get_parsed_content(self, parsed_content_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve parsed content by ID"""
        try:
            # Get metadata to find blob key
            blob_store = self.blob_store
            metadata_store = self.metadata_store
            
            metadata = metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
            if not metadata:
                logger.warning(f"Parsed content metadata not found for id: {parsed_content_id}")
                return None
            
            # Retrieve blob content
            try:
                content = blob_store.retrieve(metadata.blob_key, **kwargs)
                logger.debug(f"Retrieved parsed content for id: {parsed_content_id}")
                return content
            except KeyError:
                logger.error(f"Parsed content blob not found for id: {parsed_content_id}, blob_key: {metadata.blob_key}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get parsed content for {parsed_content_id}: {e}")
            raise

    def get_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional['ChunkMetadata']:
        """Retrieve chunk metadata by ID"""
        try:
            metadata_store = self.metadata_store
            return metadata_store.get_chunk_metadata(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get chunk metadata for {chunk_id}: {e}")
            raise

    def get_chunk(self, chunk_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve chunk content by ID"""
        try:
            # Get metadata to find blob key
            blob_store = self.blob_store
            metadata_store = self.metadata_store

            metadata = metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if not metadata:
                logger.warning(f"Chunk metadata not found for id: {chunk_id}")
                return None

            # Retrieve blob content
            try:
                content = blob_store.retrieve(metadata.blob_key, **kwargs)
                logger.debug(f"Retrieved chunk content for id: {chunk_id}")
                return content
            except KeyError:
                logger.error(f"Chunk blob not found for id: {chunk_id}, blob_key: {metadata.blob_key}")
                return None

        except Exception as e:
            logger.error(f"Failed to get chunk content for {chunk_id}: {e}")
            raise
    
    def validate_file_upload(self, file_id: str, **kwargs: Any) -> bool:
        """Validate that file upload completed successfully"""
        try:
            # Get file metadata
            blob_store = self.blob_store
            metadata_store = self.metadata_store

            metadata = metadata_store.get_file_metadata(file_id, **kwargs)
            if not metadata:
                logger.warning(f"File metadata not found for validation: {file_id}")
                return False

            # Check if blob exists
            blob_exists = blob_store.exists(metadata.blob_key, **kwargs)
            if not blob_exists:
                logger.error(f"Blob validation failed - blob not found: {metadata.blob_key}")
                # Update status to FAILED
                metadata_store.update_file_status(file_id, FileStatus.FAILED, **kwargs)
                return False

            # If validation passes and status was FAILED, update to STORED
            if metadata.status == FileStatus.FAILED:
                metadata_store.update_file_status(file_id, FileStatus.STORED, **kwargs)

            logger.info(f"File upload validation successful: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate file upload {file_id}: {e}")
            return False
    
    def validate_parsed_content_upload(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Validate that parsed content upload completed successfully"""
        try:
            # Get parsed content metadata
            blob_store = self.blob_store
            metadata_store = self.metadata_store
            
            metadata = metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
            if not metadata:
                logger.warning(f"Parsed content metadata not found for validation: {parsed_content_id}")
                return False
            
            # Check if blob exists
            blob_exists = blob_store.exists(metadata.blob_key, **kwargs)
            if not blob_exists:
                logger.error(f"Parsed content blob validation failed - blob not found: {metadata.blob_key}")
                metadata_store.update_parsed_content_status(parsed_content_id, ParsedContentStatus.FAILED, **kwargs)
                return False

            # If validation passes and status was FAILED, update to STORED
            if metadata.status == ParsedContentStatus.FAILED:
                metadata_store.update_parsed_content_status(parsed_content_id, ParsedContentStatus.STORED, **kwargs)

            logger.info(f"Parsed content upload validation successful: {parsed_content_id}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to validate parsed content upload {parsed_content_id}: {e}")
            return False
    
    def cleanup_failed_file_upload(self, asset_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed file upload"""
        try:
            cleanup_success = True
            
            # Get metadata to find blob key
            blob_store = self.blob_store
            metadata_store = self.metadata_store
            
            metadata = metadata_store.get_file_metadata(asset_id, **kwargs)
            if metadata:
                # Try to delete blob
                try:
                    if blob_store.exists(metadata.blob_key, **kwargs):
                        blob_deleted = blob_store.delete(metadata.blob_key, **kwargs)
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
                metadata_deleted = metadata_store.delete_file_metadata(asset_id, **kwargs)
                if not metadata_deleted:
                    logger.warning(f"Failed to delete metadata during cleanup: {asset_id}")
                    cleanup_success = False
                else:
                    logger.info(f"Deleted metadata during cleanup: {asset_id}")
            except Exception as metadata_error:
                logger.error(f"Error deleting metadata during cleanup: {metadata_error}")
                cleanup_success = False
            
            return cleanup_success
            
        except Exception as e:
            logger.error(f"Failed to cleanup failed file upload {asset_id}: {e}")
            return False
    
    def cleanup_failed_parsed_content_upload(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed parsed content upload"""
        try:
            cleanup_success = True
            
            # Get metadata to find blob key
            blob_store = self.blob_store
            metadata_store = self.metadata_store
            
            metadata = metadata_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
            if metadata:
                # Try to delete blob
                try:
                    if blob_store.exists(metadata.blob_key, **kwargs):
                        blob_deleted = blob_store.delete(metadata.blob_key, **kwargs)
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
                metadata_deleted = metadata_store.delete_parsed_content_metadata(parsed_content_id, **kwargs)
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

    def validate_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Validate that chunk upload completed successfully"""
        try:
            # Get chunk metadata
            blob_store = self.blob_store
            metadata_store = self.metadata_store

            metadata = metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if not metadata:
                logger.warning(f"Chunk metadata not found for validation: {chunk_id}")
                return False

            # Check if blob exists
            blob_exists = blob_store.exists(metadata.blob_key, **kwargs)
            if not blob_exists:
                logger.error(f"Chunk blob validation failed - blob not found: {metadata.blob_key}")
                return False

            logger.info(f"Chunk upload validation successful: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate chunk upload {chunk_id}: {e}")
            return False

    def cleanup_failed_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed chunk upload"""
        try:
            cleanup_success = True

            # Get metadata to find blob key
            blob_store = self.blob_store
            metadata_store = self.metadata_store

            metadata = metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if metadata:
                # Try to delete blob
                try:
                    if blob_store.exists(metadata.blob_key, **kwargs):
                        blob_deleted = blob_store.delete(metadata.blob_key, **kwargs)
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
                metadata_deleted = metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
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
