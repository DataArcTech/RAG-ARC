"""
Base class for cache database implementations.

This module defines the abstract interface for cache databases (e.g., Redis, Memcached).
Cache databases are used for:
- Storing hot data (frequently accessed data)
- Session management
- Rate limiting
- Temporary data storage

All cache database implementations should inherit from CacheDB and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict


class CacheDB(ABC):
    """
    Abstract base class for cache database operations.
    
    Defines the interface that all cache database implementations must follow.
    Supports common cache operations like get, set, delete, and list operations.
    """

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs: Any) -> bool:
        """
        Set a key-value pair in the cache.
        
        Args:
            key: Cache key
            value: Value to store (will be serialized)
            ttl: Time to live in seconds (None = no expiration)
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get(self, key: str, **kwargs: Any) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Cached value (deserialized) or None if not found
        """
        pass

    @abstractmethod
    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        pass

    @abstractmethod
    def exists(self, key: str, **kwargs: Any) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: Cache key
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def expire(self, key: str, ttl: int, **kwargs: Any) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass

    # ==================== LIST OPERATIONS ====================

    @abstractmethod
    def lpush(self, key: str, *values: Any, **kwargs: Any) -> int:
        """
        Push values to the left (head) of a list.
        
        Args:
            key: List key
            *values: Values to push
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Length of the list after push
        """
        pass

    @abstractmethod
    def rpush(self, key: str, *values: Any, **kwargs: Any) -> int:
        """
        Push values to the right (tail) of a list.
        
        Args:
            key: List key
            *values: Values to push
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Length of the list after push
        """
        pass

    @abstractmethod
    def lrange(self, key: str, start: int, stop: int, **kwargs: Any) -> List[Any]:
        """
        Get a range of elements from a list.
        
        Args:
            key: List key
            start: Start index (0-based, inclusive)
            stop: Stop index (inclusive, -1 = end of list)
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            List of elements in the specified range
        """
        pass

    @abstractmethod
    def ltrim(self, key: str, start: int, stop: int, **kwargs: Any) -> bool:
        """
        Trim a list to the specified range.
        
        Args:
            key: List key
            start: Start index (0-based, inclusive)
            stop: Stop index (inclusive)
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def llen(self, key: str, **kwargs: Any) -> int:
        """
        Get the length of a list.
        
        Args:
            key: List key
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Length of the list (0 if key doesn't exist)
        """
        pass

    # ==================== HASH OPERATIONS ====================

    @abstractmethod
    def hset(self, key: str, field: str, value: Any, **kwargs: Any) -> bool:
        """
        Set a field in a hash.
        
        Args:
            key: Hash key
            field: Field name
            value: Value to store
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if field is new, False if field was updated
        """
        pass

    @abstractmethod
    def hget(self, key: str, field: str, **kwargs: Any) -> Optional[Any]:
        """
        Get a field from a hash.
        
        Args:
            key: Hash key
            field: Field name
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Field value or None if not found
        """
        pass

    @abstractmethod
    def hgetall(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Get all fields and values from a hash.
        
        Args:
            key: Hash key
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Dictionary of field-value pairs
        """
        pass

    @abstractmethod
    def hdel(self, key: str, *fields: str, **kwargs: Any) -> int:
        """
        Delete fields from a hash.
        
        Args:
            key: Hash key
            *fields: Field names to delete
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Number of fields deleted
        """
        pass

    # ==================== UTILITY OPERATIONS ====================

    @abstractmethod
    def ping(self, **kwargs: Any) -> bool:
        """
        Test connection to cache database.
        
        Args:
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if connection is alive, False otherwise
        """
        pass

    @abstractmethod
    def flushdb(self, **kwargs: Any) -> bool:
        """
        Clear all keys in the current database.
        
        WARNING: This is a destructive operation!
        
        Args:
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass

