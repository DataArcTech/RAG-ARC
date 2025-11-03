from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
    Dict,
    List,
    Tuple,
)
import asyncio
from functools import partial

from framework.module import AbstractModule

BST = TypeVar("BST", bound="FileDB")


class FileDB(AbstractModule):
    """Blob storage base class - encapsulation layer for raw object storage operations"""
    
    @abstractmethod
    def store(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, bool]:
        """Store blob data with given key
        
        Args:
            key: Unique identifier for the blob
            data: Binary data to store
            content_type: MIME type of the content
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (storage_key, was_overwritten)
            - storage_key: The actual key used for storage
            - was_overwritten: True if an existing blob was overwritten
        """
        pass

    async def astore(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, bool]:
        """Asynchronously store blob data"""
        return await self.store(key, data, content_type, **kwargs)

    @abstractmethod
    def retrieve(self, key: str, **kwargs: Any) -> bytes:
        """Retrieve blob data by key
        
        Args:
            key: Unique identifier for the blob
            **kwargs: Additional arguments
            
        Returns:
            Binary data of the blob
            
        Raises:
            KeyError: If blob with given key doesn't exist
        """
        pass

    async def aretrieve(self, key: str, **kwargs: Any) -> bytes:
        """Asynchronously retrieve blob data"""
        return await self.retrieve(key, **kwargs)

    @abstractmethod
    def delete(self, key: str, **kwargs: Any) -> bool:
        """Delete blob by key
        
        Args:
            key: Unique identifier for the blob
            **kwargs: Additional arguments
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass

    @abstractmethod
    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if blob exists
        
        Args:
            key: Unique identifier for the blob
            **kwargs: Additional arguments
            
        Returns:
            True if blob exists, False otherwise
        """
        pass

    @abstractmethod
    def list_keys(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """List blob keys with optional prefix filter
        
        Args:
            prefix: Filter blobs by key prefix
            limit: Maximum number of keys to return
            **kwargs: Additional arguments
            
        Returns:
            List of blob keys
        """
        pass

    @abstractmethod
    def generate_presigned_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
        **kwargs: Any,
    ) -> str:
        """Generate presigned URL for blob access
        
        Args:
            key: Unique identifier for the blob
            expiration_seconds: URL expiration time in seconds
            method: HTTP method (GET, PUT, POST, DELETE)
            **kwargs: Additional arguments
            
        Returns:
            Presigned URL string
        """
        pass