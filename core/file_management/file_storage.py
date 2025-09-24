from typing import (
    Any,
    Optional,
    List,
    Dict,
)
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

from encapsulation.data_model.orm_models import FileMetadata, ParsedContentMetadata, ChunkMetadata

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
    operations. Handles multi-file uploads with atomic validation and
    cleanup of failed operations.

    Key features:
    - Multi-file upload with batch processing
    - File validation and metadata verification
    - Parsed content storage linked to source files
    - Automatic cleanup on validation failures
    - Upload session tracking
    - Comprehensive error handling and reporting

    Architecture:
        Application Layer -> FileStorage (Core) -> FileStore (Encapsulation) -> Storage Implementations

    Dependencies:
        data_store: DataStore implementation (e.g., FileStore)
    """

    def __init__(self, config):
        """Initialize FileStorage with eager data store creation"""
        super().__init__(config)
        # Build data store immediately since we always need it
        self.data_store = self.config.data_store_config.build()

    def _generate_upload_session_id(self) -> str:
        """Generate unique upload session ID"""
        import uuid
        return str(uuid.uuid4())
    
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
    
    def upload_file(
        self,
        filename: str,
        file_data: bytes,
        content_type: Optional[str] = None,
        validate_after_store: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Upload a single file with validation.

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

            # Store file
            file_metadata = self.data_store.store_file(
                file_data=file_data,
                filename=filename,
                content_type=content_type,
                **kwargs
            )

            logger.info(f"Stored file successfully: {filename} (file_id: {file_metadata.file_id})")

            # Validate after storage if requested
            if validate_after_store:
                validation_passed = self.data_store.validate_file_upload(file_metadata.file_id, **kwargs)

                if not validation_passed:
                    # Validation failed, cleanup
                    logger.error(f"File validation failed: {filename}")
                    cleanup_success = self.data_store.cleanup_failed_file_upload(file_metadata.file_id, **kwargs)
                    if cleanup_success:
                        logger.info(f"Cleaned up failed upload: {file_metadata.file_id}")
                    else:
                        logger.warning(f"Failed to cleanup after validation failure: {file_metadata.file_id}")

                    raise StorageOperationError("File validation failed after storage")

                logger.info(f"File validation passed: {filename}")

            return file_metadata.file_id

        except FileValidationError:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            logger.error(f"Storage error for {filename}: {e}")
            raise StorageOperationError(f"Storage error: {str(e)}")
    
    def upload_multiple_files(
        self,
        file_uploads: List[Dict[str, Any]],
        validate_after_store: bool = True,
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Upload multiple files with batch validation.

        Args:
            file_uploads: List of file upload dictionaries, each containing:
                - filename: str
                - file_data: bytes
                - content_type: Optional[str]
            validate_after_store: Whether to validate each file after storing
            fail_fast: Whether to stop on first failure
            **kwargs: Additional arguments

        Returns:
            Dictionary with upload results:
            - status: "success" | "failed" | "partial_success"
            - total_files: int
            - successful_uploads: int
            - failed_uploads: int
            - results: List[Dict] with individual file results
            - upload_session_id: str
            - timestamp: datetime
        """
        upload_session_id = self._generate_upload_session_id()
        timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

        logger.info(f"Starting multi-file upload session: {upload_session_id} ({len(file_uploads)} files)")

        results = []
        successful_uploads = 0
        failed_uploads = 0

        for i, file_upload in enumerate(file_uploads):
            filename = file_upload["filename"]
            logger.info(f"Processing file {i+1}/{len(file_uploads)}: {filename}")

            try:
                # Upload individual file
                file_id = self.upload_file(
                    filename=filename,
                    file_data=file_upload["file_data"],
                    content_type=file_upload.get("content_type"),
                    validate_after_store=validate_after_store,
                    **kwargs
                )

                # Get file metadata for the result
                file_metadata = self.get_file_metadata(file_id)

                # Success result
                results.append({
                    "filename": filename,
                    "success": True,
                    "file_id": file_id,
                    "file_metadata": file_metadata,
                    "error_message": None
                })
                successful_uploads += 1
                logger.info(f"Successfully uploaded: {filename}")

            except (FileValidationError, StorageOperationError) as e:
                # Error result
                results.append({
                    "filename": filename,
                    "success": False,
                    "file_id": None,
                    "file_metadata": None,
                    "error_message": str(e)
                })
                failed_uploads += 1
                logger.error(f"Failed to upload: {filename} - {str(e)}")

                # Stop processing if fail_fast is enabled
                if fail_fast:
                    logger.warning(f"Stopping upload session due to failure (fail_fast enabled)")
                    break

        # Determine overall status
        if successful_uploads == len(file_uploads):
            overall_status = "success"
        elif successful_uploads == 0:
            overall_status = "failed"
        else:
            overall_status = "partial_success"

        success_rate = (successful_uploads / len(file_uploads)) * 100 if file_uploads else 0.0

        upload_result = {
            "status": overall_status,
            "total_files": len(file_uploads),
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads,
            "success_rate": success_rate,
            "results": results,
            "upload_session_id": upload_session_id,
            "timestamp": timestamp
        }

        logger.info(f"Completed upload session {upload_session_id}: "
                   f"{successful_uploads}/{len(file_uploads)} successful "
                   f"({success_rate:.1f}% success rate)")

        return upload_result
    
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
        Store parsed content linked to source file.

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
            # Get data store
            data_store = self.data_store

            # Store parsed content
            parsed_metadata = data_store.store_parsed_content(
                parsed_data=parsed_data,
                source_file_id=source_file_id,
                parser_type=parser_type,
                content_type=content_type,
                **kwargs
            )

            logger.info(f"Stored parsed content successfully: {parsed_metadata.parsed_content_id}")

            # Validate after storage if requested
            if validate_after_store:
                validation_passed = data_store.validate_parsed_content_upload(
                    parsed_metadata.parsed_content_id, **kwargs
                )

                if not validation_passed:
                    # Validation failed, cleanup
                    logger.error(f"Parsed content validation failed: {parsed_metadata.parsed_content_id}")
                    cleanup_success = data_store.cleanup_failed_parsed_content_upload(
                        parsed_metadata.parsed_content_id, **kwargs
                    )
                    if cleanup_success:
                        logger.info(f"Cleaned up failed parsed content: {parsed_metadata.parsed_content_id}")
                    else:
                        logger.warning(f"Failed to cleanup after validation failure: {parsed_metadata.parsed_content_id}")

                    raise StorageOperationError("Parsed content validation failed after storage")

                logger.info(f"Parsed content validation passed: {parsed_metadata.parsed_content_id}")

            return parsed_metadata.parsed_content_id

        except FileValidationError:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            logger.error(f"Parsed content storage error: {e}")
            raise StorageOperationError(f"Storage error: {str(e)}")
    
    def store_multiple_parsed_content(
        self,
        parsed_content_list: List[Dict[str, Any]],
        validate_after_store: bool = True,
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Store multiple parsed content items with batch processing.

        Args:
            parsed_content_list: List of parsed content dictionaries, each containing:
                - source_file_id: str
                - parser_type: str
                - parsed_data: bytes
                - content_type: str (default: "text/markdown")
            validate_after_store: Whether to validate each parsed content after storing
            fail_fast: Whether to stop on first failure
            **kwargs: Additional arguments

        Returns:
            Dictionary with storage results:
            - status: "success" | "failed" | "partial_success"
            - total_contents: int
            - successful_storages: int
            - failed_storages: int
            - results: List[Dict] with individual content results
            - processing_session_id: str
            - timestamp: datetime
        """
        processing_session_id = self._generate_upload_session_id()
        timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

        logger.info(f"Starting multi-parsed content storage session: {processing_session_id} ({len(parsed_content_list)} contents)")

        results = []
        successful_storages = 0
        failed_storages = 0

        for i, parsed_content in enumerate(parsed_content_list):
            source_file_id = parsed_content["source_file_id"]
            logger.info(f"Processing parsed content {i+1}/{len(parsed_content_list)} for source: {source_file_id}")

            try:
                # Store individual parsed content
                parsed_content_id = self.store_parsed_content(
                    source_file_id=source_file_id,
                    parser_type=parsed_content["parser_type"],
                    parsed_data=parsed_content["parsed_data"],
                    content_type=parsed_content.get("content_type", "text/markdown"),
                    validate_after_store=validate_after_store,
                    **kwargs
                )

                # Get parsed content metadata for the result
                parsed_metadata = self.get_parsed_content_metadata(parsed_content_id)

                # Success result
                results.append({
                    "source_file_id": source_file_id,
                    "success": True,
                    "parsed_content_id": parsed_content_id,
                    "parsed_metadata": parsed_metadata,
                    "error_message": None
                })
                successful_storages += 1
                logger.info(f"Successfully stored parsed content: {parsed_content_id}")

            except (FileValidationError, StorageOperationError) as e:
                # Error result
                results.append({
                    "source_file_id": source_file_id,
                    "success": False,
                    "parsed_content_id": None,
                    "parsed_metadata": None,
                    "error_message": str(e)
                })
                failed_storages += 1
                logger.error(f"Failed to store parsed content for source {source_file_id}: {str(e)}")

                # Stop processing if fail_fast is enabled
                if fail_fast:
                    logger.warning(f"Stopping parsed content storage session due to failure (fail_fast enabled)")
                    break

        # Determine overall status
        if successful_storages == len(parsed_content_list):
            overall_status = "success"
        elif successful_storages == 0:
            overall_status = "failed"
        else:
            overall_status = "partial_success"

        success_rate = (successful_storages / len(parsed_content_list)) * 100 if parsed_content_list else 0.0

        storage_result = {
            "status": overall_status,
            "total_contents": len(parsed_content_list),
            "successful_storages": successful_storages,
            "failed_storages": failed_storages,
            "success_rate": success_rate,
            "results": results,
            "processing_session_id": processing_session_id,
            "timestamp": timestamp
        }

        logger.info(f"Completed parsed content storage session {processing_session_id}: "
                   f"{successful_storages}/{len(parsed_content_list)} successful "
                   f"({success_rate:.1f}% success rate)")

        return storage_result

    def store_chunk(
        self,
        source_parsed_content_id: str,
        chunker_type: str,
        chunk_data: bytes,
        validate_after_store: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Store a single chunk linked to parsed content.

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
            # Get data store
            data_store = self.data_store

            # Store chunk
            chunk_metadata = data_store.store_chunk(
                chunk_data=chunk_data,
                source_parsed_content_id=source_parsed_content_id,
                chunker_type=chunker_type,
                **kwargs
            )

            logger.info(f"Stored chunk successfully: {chunk_metadata.chunk_id}")

            # Validate after storage if requested
            if validate_after_store:
                validation_passed = data_store.validate_chunk_upload(
                    chunk_metadata.chunk_id, **kwargs
                )

                if not validation_passed:
                    # Validation failed, cleanup
                    logger.error(f"Chunk validation failed: {chunk_metadata.chunk_id}")
                    cleanup_success = data_store.cleanup_failed_chunk_upload(
                        chunk_metadata.chunk_id, **kwargs
                    )
                    if cleanup_success:
                        logger.info(f"Cleaned up failed chunk: {chunk_metadata.chunk_id}")
                    else:
                        logger.warning(f"Failed to cleanup after validation failure: {chunk_metadata.chunk_id}")

                    raise StorageOperationError("Chunk validation failed after storage")

                logger.info(f"Chunk validation passed: {chunk_metadata.chunk_id}")

            return chunk_metadata.chunk_id

        except FileValidationError:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            logger.error(f"Chunk storage error: {e}")
            raise StorageOperationError(f"Storage error: {str(e)}")

    def store_multiple_chunks(
        self,
        chunks_list: List[Dict[str, Any]],
        validate_after_store: bool = True,
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Store multiple individual chunks with batch processing.

        Args:
            chunks_list: List of chunk dictionaries, each containing:
                - source_parsed_content_id: str
                - chunker_type: str
                - chunk_data: bytes (JSON format)
            validate_after_store: Whether to validate each chunk after storing
            fail_fast: Whether to stop on first failure
            **kwargs: Additional arguments

        Returns:
            Dictionary with storage results:
            - status: "success" | "failed" | "partial_success"
            - total_chunks: int
            - successful_storages: int
            - failed_storages: int
            - results: List[Dict] with individual chunk results
            - processing_session_id: str
            - timestamp: datetime
        """
        processing_session_id = self._generate_upload_session_id()
        timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

        logger.info(f"Starting multi-chunk storage session: {processing_session_id} ({len(chunks_list)} chunks)")

        results = []
        successful_storages = 0
        failed_storages = 0

        for i, chunk_item in enumerate(chunks_list):
            source_parsed_content_id = chunk_item["source_parsed_content_id"]
            logger.info(f"Processing chunk {i+1}/{len(chunks_list)} for source: {source_parsed_content_id}")

            try:
                # Store individual chunk
                chunk_id = self.store_chunk(
                    source_parsed_content_id=source_parsed_content_id,
                    chunker_type=chunk_item["chunker_type"],
                    chunk_data=chunk_item["chunk_data"],
                    validate_after_store=validate_after_store,
                    **kwargs
                )

                # Get chunk metadata for the result
                chunk_metadata = self.get_chunk_metadata(chunk_id)

                # Success result
                results.append({
                    "source_parsed_content_id": source_parsed_content_id,
                    "success": True,
                    "chunk_id": chunk_id,
                    "chunk_metadata": chunk_metadata,
                    "error_message": None
                })
                successful_storages += 1
                logger.info(f"Successfully stored chunk: {chunk_id}")

            except (FileValidationError, StorageOperationError) as e:
                # Error result
                results.append({
                    "source_parsed_content_id": source_parsed_content_id,
                    "success": False,
                    "chunk_id": None,
                    "chunk_metadata": None,
                    "error_message": str(e)
                })
                failed_storages += 1
                logger.error(f"Failed to store chunk for source {source_parsed_content_id}: {str(e)}")

                # Stop processing if fail_fast is enabled
                if fail_fast:
                    logger.warning(f"Stopping chunk storage session due to failure (fail_fast enabled)")
                    break

        # Determine overall status
        if successful_storages == len(chunks_list):
            overall_status = "success"
        elif successful_storages == 0:
            overall_status = "failed"
        else:
            overall_status = "partial_success"

        success_rate = (successful_storages / len(chunks_list)) * 100 if chunks_list else 0.0

        storage_result = {
            "status": overall_status,
            "total_chunks": len(chunks_list),
            "successful_storages": successful_storages,
            "failed_storages": failed_storages,
            "success_rate": success_rate,
            "results": results,
            "processing_session_id": processing_session_id,
            "timestamp": timestamp
        }

        logger.info(f"Completed chunk storage session {processing_session_id}: "
                   f"{successful_storages}/{len(chunks_list)} successful "
                   f"({success_rate:.1f}% success rate)")

        return storage_result

    
    def get_file_metadata(self, file_id: str, **kwargs: Any) -> Optional['FileMetadata']:
        """Retrieve file metadata by file ID"""
        try:
            data_store = self.data_store
            return data_store.get_file_metadata(file_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get file metadata for {file_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve file metadata: {e}")

    def get_file_content(self, file_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve file content by file ID"""
        try:
            data_store = self.data_store
            return data_store.get_file_content(file_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get file content for {file_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve file content: {e}")
    
    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional['ParsedContentMetadata']:
        """Retrieve parsed content metadata by ID"""
        try:
            data_store = self.data_store
            return data_store.get_parsed_content_metadata(parsed_content_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get parsed content metadata for {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve parsed content metadata: {e}")
    
    def get_parsed_content(self, parsed_content_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve parsed content by ID"""
        try:
            data_store = self.data_store
            return data_store.get_parsed_content(parsed_content_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get parsed content for {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve parsed content: {e}")

    def get_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional['ChunkMetadata']:
        """Retrieve chunk metadata by ID"""
        try:
            data_store = self.data_store
            return data_store.get_chunk_metadata(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get chunk metadata for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve chunk metadata: {e}")

    def get_chunk_content(self, chunk_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve chunk content by ID"""
        try:
            data_store = self.data_store
            return data_store.get_chunk(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get chunk content for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve chunk content: {e}")
    
    def delete_file(self, file_id: str, **kwargs: Any) -> bool:
        """Delete file and cleanup associated data"""
        try:
            data_store = self.data_store
            return data_store.cleanup_failed_file_upload(file_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise StorageOperationError(f"Failed to delete file: {e}")
    
    def delete_parsed_content(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Delete parsed content and cleanup associated data"""
        try:
            data_store = self.data_store
            return data_store.cleanup_failed_parsed_content_upload(parsed_content_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete parsed content {parsed_content_id}: {e}")
            raise StorageOperationError(f"Failed to delete parsed content: {e}")

    def delete_chunk(self, chunk_id: str, **kwargs: Any) -> bool:
        """Delete chunk and cleanup associated data"""
        try:
            data_store = self.data_store
            return data_store.cleanup_failed_chunk_upload(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to delete chunk: {e}")