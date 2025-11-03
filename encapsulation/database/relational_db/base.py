from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
    List,
    Dict,
)
import asyncio
from functools import partial

if TYPE_CHECKING:
    from ...data_model.orm_models import FileMetadata, FileStatus
    from ...data_model.orm_models import ParsedContentMetadata, ParsedContentStatus
    from ...data_model.orm_models import ChunkMetadata, ChunkIndexStatus

from framework.module import AbstractModule

MST = TypeVar("MST", bound="RelationalDB")


class RelationalDB(AbstractModule):
    """Metadata storage base class - encapsulation layer for file metadata operations"""
    
    @abstractmethod
    def store_file_metadata(
        self,
        file_metadata: 'FileMetadata',
        **kwargs: Any,
    ) -> str:
        """Store file metadata

        Args:
            file_metadata: FileMetadata object to store
            **kwargs: Additional arguments

        Returns:
            File ID of stored metadata
        """
        pass

    @abstractmethod
    def get_file_metadata(self, file_id: str, **kwargs: Any) -> Optional['FileMetadata']:
        """Retrieve file metadata by file ID

        Args:
            file_id: Unique identifier for the file
            **kwargs: Additional arguments

        Returns:
            FileMetadata object if found, None otherwise
        """
        pass

    @abstractmethod
    def update_file_metadata(
        self,
        file_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update file metadata

        Args:
            file_id: Unique identifier for the file
            updates: Dictionary of fields to update
            **kwargs: Additional arguments

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_file_metadata(self, file_id: str, **kwargs: Any) -> bool:
        """Delete file metadata

        Args:
            file_id: Unique identifier for the file
            **kwargs: Additional arguments

        Returns:
            True if deletion successful, False otherwise
        """
        pass

    @abstractmethod
    def update_file_status(
        self,
        file_id: str,
        new_status: 'FileStatus',
        **kwargs: Any,
    ) -> bool:
        """Update file processing status

        Args:
            file_id: Unique identifier for the file
            new_status: New processing status
            **kwargs: Additional arguments

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def list_file_metadata(
        self,
        status: Optional['FileStatus'] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List['FileMetadata']:
        """List file metadata with optional filtering

        Args:
            status: Filter by file status
            limit: Maximum number of records to return
            offset: Number of records to skip
            **kwargs: Additional arguments

        Returns:
            List of FileMetadata objects
        """
        pass

    # ==================== PARSED CONTENT METADATA METHODS ====================

    @abstractmethod
    def store_parsed_content_metadata(
        self,
        parsed_content_metadata: 'ParsedContentMetadata',
        **kwargs: Any,
    ) -> str:
        """Store parsed content metadata

        Args:
            parsed_content_metadata: ParsedContentMetadata object to store
            **kwargs: Additional arguments

        Returns:
            Parsed content ID of stored metadata
        """
        pass

    @abstractmethod
    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional['ParsedContentMetadata']:
        """Retrieve parsed content metadata by ID

        Args:
            parsed_content_id: Unique identifier for the parsed content
            **kwargs: Additional arguments

        Returns:
            ParsedContentMetadata object if found, None otherwise
        """
        pass

    @abstractmethod
    def update_parsed_content_metadata(
        self,
        parsed_content_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update parsed content metadata

        Args:
            parsed_content_id: Unique identifier for the parsed content
            updates: Dictionary of fields to update
            **kwargs: Additional arguments

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Delete parsed content metadata

        Args:
            parsed_content_id: Unique identifier for the parsed content
            **kwargs: Additional arguments

        Returns:
            True if deletion successful, False otherwise
        """
        pass

    @abstractmethod
    def update_parsed_content_status(
        self,
        parsed_content_id: str,
        new_status: 'ParsedContentStatus',
        **kwargs: Any,
    ) -> bool:
        """Update parsed content processing status

        Args:
            parsed_content_id: Unique identifier for the parsed content
            new_status: New processing status
            **kwargs: Additional arguments

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def list_parsed_content_metadata(
        self,
        source_file_id: Optional[str] = None,
        status: Optional['ParsedContentStatus'] = None,
        parser_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List['ParsedContentMetadata']:
        """List parsed content metadata with optional filtering

        Args:
            source_file_id: Filter by source file ID
            status: Filter by parsed content status
            parser_type: Filter by parser type
            limit: Maximum number of records to return
            offset: Number of records to skip
            **kwargs: Additional arguments

        Returns:
            List of ParsedContentMetadata objects
        """
        pass

    # ==================== CHUNK METADATA METHODS ====================

    @abstractmethod
    def store_chunk_metadata(
        self,
        chunk_metadata: 'ChunkMetadata',
        **kwargs: Any,
    ) -> str:
        """Store chunk metadata

        Args:
            chunk_metadata: ChunkMetadata object to store
            **kwargs: Additional arguments

        Returns:
            Chunk ID of stored metadata
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
    def update_chunk_metadata(
        self,
        chunk_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update chunk metadata

        Args:
            chunk_id: Unique identifier for the chunk
            updates: Dictionary of fields to update
            **kwargs: Additional arguments

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> bool:
        """Delete chunk metadata

        Args:
            chunk_id: Unique identifier for the chunk
            **kwargs: Additional arguments

        Returns:
            True if deletion successful, False otherwise
        """
        pass

    @abstractmethod
    def list_chunk_metadata(
        self,
        source_parsed_content_id: Optional[str] = None,
        index_status: Optional['ChunkIndexStatus'] = None,
        chunker_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List['ChunkMetadata']:
        """List chunk metadata with optional filtering

        Args:
            source_parsed_content_id: Filter by source parsed content ID
            index_status: Filter by chunk index status
            chunker_type: Filter by chunker type
            limit: Maximum number of records to return
            offset: Number of records to skip
            **kwargs: Additional arguments

        Returns:
            List of ChunkMetadata objects
        """
        pass

    # ==================== ASYNC METHODS ====================

    async def astore_file_metadata(
        self,
        file_metadata: 'FileMetadata',
        **kwargs: Any,
    ) -> str:
        """Asynchronously store file metadata"""
        return await self.store_file_metadata(file_metadata, **kwargs)

    async def aget_file_metadata(self, file_id: str, **kwargs: Any) -> Optional['FileMetadata']:
        """Asynchronously retrieve file metadata"""
        return await self.get_file_metadata(file_id, **kwargs)

    async def astore_parsed_content_metadata(
        self,
        parsed_content_metadata: 'ParsedContentMetadata',
        **kwargs: Any,
    ) -> str:
        """Asynchronously store parsed content metadata"""
        return await self.store_parsed_content_metadata(parsed_content_metadata, **kwargs)

    async def aget_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional['ParsedContentMetadata']:
        """Asynchronously retrieve parsed content metadata"""
        return await self.get_parsed_content_metadata(parsed_content_id, **kwargs)

    async def astore_chunk_metadata(
        self,
        chunk_metadata: 'ChunkMetadata',
        **kwargs: Any,
    ) -> str:
        """Asynchronously store chunk metadata"""
        return await self.store_chunk_metadata(chunk_metadata, **kwargs)

    async def aget_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional['ChunkMetadata']:
        """Asynchronously retrieve chunk metadata"""
        return await self.get_chunk_metadata(chunk_id, **kwargs)