from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
    BinaryIO,
)
import asyncio
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from ..data_model.orm_models import FileMetadata
    from ..data_model.orm_models import ParsedContentMetadata
    from ..data_model.orm_models import ChunkMetadata

from framework.module import AbstractModule

FDB = TypeVar("FDB", bound="DataStore")


class DataStore(AbstractModule):
    """File database base class - encapsulation layer for coordinated file and metadata operations"""
    
    @abstractmethod
    def store_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None,
        original_path: Optional[str] = None,
        **kwargs: Any,
    ) -> 'FileMetadata':
        """Store file data and create metadata record
        
        This method coordinates:
        1. Storing file data to blob storage (S3/MinIO/Local)
        2. Creating and storing metadata record
        
        Args:
            file_data: Binary file data to store
            filename: Original filename
            content_type: MIME type of the file
            original_path: Original file path if uploaded from filesystem
            **kwargs: Additional arguments
            
        Returns:
            FileMetadata object with storage details
        """
        pass

    async def astore_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None,
        original_path: Optional[str] = None,
        **kwargs: Any,
    ) -> 'FileMetadata':
        """Asynchronously store file data and metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.store_file, file_data, filename, content_type, original_path, **kwargs
        )

    @abstractmethod
    def store_parsed_content(
        self,
        parsed_data: bytes,
        source_file_id: str,
        parser_type: str,
        parser_version: Optional[str] = None,
        content_type: str = "text/markdown",
        parsing_config: Optional[str] = None,
        page_count: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> 'ParsedContentMetadata':
        """Store parsed content data and create metadata record
        
        This method coordinates:
        1. Storing parsed content data to blob storage
        2. Creating and storing parsed content metadata record
        
        Args:
            parsed_data: Binary parsed content data (markdown, text, etc.)
            source_file_id: ID of the original file that was parsed
            parser_type: Type of parser used (e.g., "dots_ocr", "pypdf")
            parser_version: Version of the parser
            content_type: MIME type of parsed content (default: "text/markdown")
            parsing_config: JSON string of parser configuration used
            page_count: Number of pages/sections parsed
            language: Detected language of content
            **kwargs: Additional arguments
            
        Returns:
            ParsedContentMetadata object with storage details
        """
        pass

    async def astore_parsed_content(
        self,
        parsed_data: bytes,
        source_file_id: str,
        parser_type: str,
        parser_version: str,
        content_type: str = "text/markdown",
        parsing_config: Optional[str] = None,
        page_count: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> 'ParsedContentMetadata':
        """Asynchronously store parsed content data and metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.store_parsed_content, parsed_data, source_file_id,
            parser_type, parser_version, content_type, parsing_config, page_count, language, **kwargs
        )

    @abstractmethod
    def validate_file_upload(self, file_id: str, **kwargs: Any) -> bool:
        """Validate that file upload completed successfully

        Checks both blob storage and metadata storage to ensure consistency.
        Updates file status to UPLOADED or FAILED based on validation.

        Args:
            file_id: Unique identifier for the file
            **kwargs: Additional arguments

        Returns:
            True if upload is valid and complete, False otherwise
        """
        pass

    @abstractmethod
    def validate_parsed_content_upload(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Validate that parsed content upload completed successfully
        
        Args:
            parsed_content_id: Unique identifier for the parsed content
            **kwargs: Additional arguments
            
        Returns:
            True if upload is valid and complete, False otherwise
        """
        pass

    @abstractmethod
    def cleanup_failed_file_upload(self, file_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed file upload

        Removes partial uploads from both blob storage and metadata storage.

        Args:
            file_id: Unique identifier for the failed file upload
            **kwargs: Additional arguments

        Returns:
            True if cleanup successful, False otherwise
        """
        pass

    @abstractmethod
    def cleanup_failed_parsed_content_upload(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed parsed content upload
        
        Args:
            parsed_content_id: Unique identifier for the failed parsed content
            **kwargs: Additional arguments

        Returns:
            True if cleanup successful, False otherwise
        """
        pass

    @abstractmethod
    def store_chunk(
        self,
        chunk_data: bytes,
        source_parsed_content_id: str,
        chunker_type: str,
        **kwargs: Any,
    ) -> 'ChunkMetadata':
        """Store individual chunk data and create metadata record

        Args:
            chunk_data: Binary chunk data (JSON format)
            source_parsed_content_id: ID of the parsed content this chunk comes from
            chunker_type: Type of chunker used (e.g., "semantic", "token")
            **kwargs: Additional arguments

        Returns:
            ChunkMetadata object with storage details
        """
        pass

    @abstractmethod
    def get_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional['ChunkMetadata']:
        """Retrieve chunk metadata by ID

        Args:
            chunk_id: Unique identifier for the chunk
            **kwargs: Additional arguments

        Returns:
            ChunkMetadata object if found, None otherwise
        """
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve chunk content by ID

        Args:
            chunk_id: Unique identifier for the chunk
            **kwargs: Additional arguments

        Returns:
            Chunk content as bytes if found, None otherwise
        """
        pass

    @abstractmethod
    def validate_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Validate that chunk upload completed successfully

        Args:
            chunk_id: Unique identifier for the chunk
            **kwargs: Additional arguments

        Returns:
            True if upload is valid and complete, False otherwise
        """
        pass

    @abstractmethod
    def cleanup_failed_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed chunk upload

        Args:
            chunk_id: Unique identifier for the failed chunk upload
            **kwargs: Additional arguments

        Returns:
            True if cleanup successful, False otherwise
        """
        pass

    async def avalidate_file_upload(self, file_id: str, **kwargs: Any) -> bool:
        """Asynchronously validate file upload"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.validate_file_upload, file_id, **kwargs
        )

    async def acleanup_failed_file_upload(self, file_id: str, **kwargs: Any) -> bool:
        """Asynchronously cleanup failed file upload"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.cleanup_failed_file_upload, file_id, **kwargs
        )

    async def avalidate_parsed_content_upload(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Asynchronously validate parsed content upload"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.validate_parsed_content_upload, parsed_content_id, **kwargs
        )

    async def acleanup_failed_parsed_content_upload(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Asynchronously cleanup failed parsed content upload"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.cleanup_failed_parsed_content_upload, parsed_content_id, **kwargs
        )

    async def avalidate_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Asynchronously validate chunk upload"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.validate_chunk_upload, chunk_id, **kwargs
        )

    async def acleanup_failed_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Asynchronously cleanup failed chunk upload"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.cleanup_failed_chunk_upload, chunk_id, **kwargs
        )

  