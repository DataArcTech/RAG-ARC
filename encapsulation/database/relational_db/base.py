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
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from .data_schema import FileMetadata, FileStatus
    from .data_schema import ParsedContentMetadata, ParsedContentStatus
    from .data_schema import ChunksMetadata, ChunksStatus

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
            Asset ID of stored metadata
        """
        pass

    @abstractmethod
    def get_file_metadata(self, asset_id: str, **kwargs: Any) -> Optional['FileMetadata']:
        """Retrieve file metadata by asset ID
        
        Args:
            asset_id: Unique identifier for the file asset
            **kwargs: Additional arguments
            
        Returns:
            FileMetadata object if found, None otherwise
        """
        pass

    @abstractmethod
    def update_file_metadata(
        self,
        asset_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update file metadata
        
        Args:
            asset_id: Unique identifier for the file asset
            updates: Dictionary of fields to update
            **kwargs: Additional arguments
            
        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_file_metadata(self, asset_id: str, **kwargs: Any) -> bool:
        """Delete file metadata
        
        Args:
            asset_id: Unique identifier for the file asset
            **kwargs: Additional arguments
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass

    @abstractmethod
    def update_file_status(
        self,
        asset_id: str,
        new_status: 'FileStatus',
        **kwargs: Any,
    ) -> bool:
        """Update file processing status
        
        Args:
            asset_id: Unique identifier for the file asset
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
        source_asset_id: Optional[str] = None,
        status: Optional['ParsedContentStatus'] = None,
        parser_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List['ParsedContentMetadata']:
        """List parsed content metadata with optional filtering

        Args:
            source_asset_id: Filter by source file asset ID
            status: Filter by parsed content status
            parser_type: Filter by parser type
            limit: Maximum number of records to return
            offset: Number of records to skip
            **kwargs: Additional arguments

        Returns:
            List of ParsedContentMetadata objects
        """
        pass

    # ==================== CHUNKS METADATA METHODS ====================

    @abstractmethod
    def store_chunks_metadata(
        self,
        chunks_metadata: 'ChunksMetadata',
        **kwargs: Any,
    ) -> str:
        """Store chunks metadata

        Args:
            chunks_metadata: ChunksMetadata object to store
            **kwargs: Additional arguments

        Returns:
            Chunks ID of stored metadata
        """
        pass

    @abstractmethod
    def get_chunks_metadata(self, chunks_id: str, **kwargs: Any) -> Optional['ChunksMetadata']:
        """Retrieve chunks metadata by ID

        Args:
            chunks_id: Unique identifier for the chunks
            **kwargs: Additional arguments

        Returns:
            ChunksMetadata object if found, None otherwise
        """
        pass

    @abstractmethod
    def update_chunks_metadata(
        self,
        chunks_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update chunks metadata

        Args:
            chunks_id: Unique identifier for the chunks
            updates: Dictionary of fields to update
            **kwargs: Additional arguments

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_chunks_metadata(self, chunks_id: str, **kwargs: Any) -> bool:
        """Delete chunks metadata

        Args:
            chunks_id: Unique identifier for the chunks
            **kwargs: Additional arguments

        Returns:
            True if deletion successful, False otherwise
        """
        pass

    @abstractmethod
    def update_chunks_status(
        self,
        chunks_id: str,
        new_status: 'ChunksStatus',
        **kwargs: Any,
    ) -> bool:
        """Update chunks processing status

        Args:
            chunks_id: Unique identifier for the chunks
            new_status: New processing status
            **kwargs: Additional arguments

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    def list_chunks_metadata(
        self,
        source_parsed_content_id: Optional[str] = None,
        status: Optional['ChunksStatus'] = None,
        chunking_strategy: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List['ChunksMetadata']:
        """List chunks metadata with optional filtering

        Args:
            source_parsed_content_id: Filter by source parsed content ID
            status: Filter by chunks status
            chunking_strategy: Filter by chunking strategy
            limit: Maximum number of records to return
            offset: Number of records to skip
            **kwargs: Additional arguments

        Returns:
            List of ChunksMetadata objects
        """
        pass

    # ==================== ASYNC METHODS ====================

    async def astore_file_metadata(
        self,
        file_metadata: 'FileMetadata',
        **kwargs: Any,
    ) -> str:
        """Asynchronously store file metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.store_file_metadata, file_metadata, **kwargs
        )

    async def aget_file_metadata(self, asset_id: str, **kwargs: Any) -> Optional['FileMetadata']:
        """Asynchronously retrieve file metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.get_file_metadata, asset_id, **kwargs
        )

    async def astore_parsed_content_metadata(
        self,
        parsed_content_metadata: 'ParsedContentMetadata',
        **kwargs: Any,
    ) -> str:
        """Asynchronously store parsed content metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.store_parsed_content_metadata, parsed_content_metadata, **kwargs
        )

    async def aget_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional['ParsedContentMetadata']:
        """Asynchronously retrieve parsed content metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.get_parsed_content_metadata, parsed_content_id, **kwargs
        )

    async def astore_chunks_metadata(
        self,
        chunks_metadata: 'ChunksMetadata',
        **kwargs: Any,
    ) -> str:
        """Asynchronously store chunks metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.store_chunks_metadata, chunks_metadata, **kwargs
        )

    async def aget_chunks_metadata(self, chunks_id: str, **kwargs: Any) -> Optional['ChunksMetadata']:
        """Asynchronously retrieve chunks metadata"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.get_chunks_metadata, chunks_id, **kwargs
        )