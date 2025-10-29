"""
Redis implementation for cache database.

This module provides a Redis-based cache database implementation with:
- Connection pooling for high performance
- Automatic JSON serialization/deserialization
- Support for string, list, and hash operations
- Configurable persistence (RDB + AOF)
- Error handling and logging

Redis is used for:
- Hot data caching (e.g., recent chat messages)
- Session management
- Rate limiting
- Temporary data storage

Configuration:
    host: Redis server hostname
    port: Redis server port (default: "6379")
    db: Redis database number (default: 0)
    password: Redis password (optional)
    max_connections: Maximum connections in pool (default: 50)
    decode_responses: Decode responses to strings (default: True)
"""

from typing import Any, Optional, List, Dict, TYPE_CHECKING
import logging
import json
import redis
from redis.connection import ConnectionPool

from .base import CacheDB
from framework.singleton_decorator import singleton

if TYPE_CHECKING:
    from config.encapsulation.database.cache_db.redis_config import RedisConfig

logger = logging.getLogger(__name__)


@singleton
class RedisDB(CacheDB):
    """
    Redis implementation for cache database operations.
    
    Features:
    - Connection pooling for performance
    - Automatic JSON serialization for complex objects
    - Support for TTL (time to live)
    - List and hash operations for structured data
    - Persistence with RDB + AOF (configured on Redis server)
    
    Usage:
        >>> config = RedisConfig(host="localhost", port="6379", db="0")
        >>> redis_db = RedisDB(config)
        >>> redis_db.set("key", {"data": "value"}, ttl=3600)
        >>> value = redis_db.get("key")
    """

    def __init__(self, config: "RedisConfig"):
        """
        Initialize Redis connection with connection pooling.
        
        Args:
            config: RedisConfig object with connection parameters
        """
        self.config = config
        
        # Create connection pool
        self.pool = ConnectionPool(
            host=config.host,
            port=int(config.port),
            db=int(config.db),
            password=config.password if config.password else None,
            max_connections=config.max_connections,
            decode_responses=config.decode_responses,
        )
        
        # Create Redis client
        self.client = redis.Redis(connection_pool=self.pool)
        
        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {config.host}:{config.port} (db={config.db})")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string"""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        return json.dumps(value, ensure_ascii=False)

    def _deserialize(self, value: Optional[str]) -> Optional[Any]:
        """Deserialize JSON string to value"""
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    # ==================== BASIC OPERATIONS ====================

    def set(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs: Any) -> bool:
        """Set a key-value pair with optional TTL"""
        try:
            serialized_value = self._serialize(value)
            if ttl:
                result = self.client.setex(key, ttl, serialized_value)
            else:
                result = self.client.set(key, serialized_value)
            
            logger.debug(f"Set key: {key} (ttl={ttl})")
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def get(self, key: str, **kwargs: Any) -> Optional[Any]:
        """Get a value by key"""
        try:
            value = self.client.get(key)
            return self._deserialize(value)
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def delete(self, key: str, **kwargs: Any) -> bool:
        """Delete a key"""
        try:
            result = self.client.delete(key)
            logger.debug(f"Deleted key: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if a key exists"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False

    def expire(self, key: str, ttl: int, **kwargs: Any) -> bool:
        """Set expiration time for a key"""
        try:
            result = self.client.expire(key, ttl)
            logger.debug(f"Set expiration for key {key}: {ttl}s")
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False

    # ==================== LIST OPERATIONS ====================

    def lpush(self, key: str, *values: Any, **kwargs: Any) -> int:
        """Push values to the left (head) of a list"""
        try:
            serialized_values = [self._serialize(v) for v in values]
            result = self.client.lpush(key, *serialized_values)
            logger.debug(f"Pushed {len(values)} values to list {key}")
            return result
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return 0

    def rpush(self, key: str, *values: Any, **kwargs: Any) -> int:
        """Push values to the right (tail) of a list"""
        try:
            serialized_values = [self._serialize(v) for v in values]
            result = self.client.rpush(key, *serialized_values)
            logger.debug(f"Pushed {len(values)} values to list {key}")
            return result
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return 0

    def lrange(self, key: str, start: int, stop: int, **kwargs: Any) -> List[Any]:
        """Get a range of elements from a list"""
        try:
            values = self.client.lrange(key, start, stop)
            return [self._deserialize(v) for v in values]
        except Exception as e:
            logger.error(f"Error getting range from list {key}: {e}")
            return []

    def ltrim(self, key: str, start: int, stop: int, **kwargs: Any) -> bool:
        """Trim a list to the specified range"""
        try:
            result = self.client.ltrim(key, start, stop)
            logger.debug(f"Trimmed list {key} to [{start}, {stop}]")
            return bool(result)
        except Exception as e:
            logger.error(f"Error trimming list {key}: {e}")
            return False

    def llen(self, key: str, **kwargs: Any) -> int:
        """Get the length of a list"""
        try:
            return self.client.llen(key)
        except Exception as e:
            logger.error(f"Error getting length of list {key}: {e}")
            return 0

    # ==================== HASH OPERATIONS ====================

    def hset(self, key: str, field: str, value: Any, **kwargs: Any) -> bool:
        """Set a field in a hash"""
        try:
            serialized_value = self._serialize(value)
            result = self.client.hset(key, field, serialized_value)
            logger.debug(f"Set hash field: {key}.{field}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting hash field {key}.{field}: {e}")
            return False

    def hget(self, key: str, field: str, **kwargs: Any) -> Optional[Any]:
        """Get a field from a hash"""
        try:
            value = self.client.hget(key, field)
            return self._deserialize(value)
        except Exception as e:
            logger.error(f"Error getting hash field {key}.{field}: {e}")
            return None

    def hgetall(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Get all fields and values from a hash"""
        try:
            hash_data = self.client.hgetall(key)
            return {k: self._deserialize(v) for k, v in hash_data.items()}
        except Exception as e:
            logger.error(f"Error getting all hash fields from {key}: {e}")
            return {}

    def hdel(self, key: str, *fields: str, **kwargs: Any) -> int:
        """Delete fields from a hash"""
        try:
            result = self.client.hdel(key, *fields)
            logger.debug(f"Deleted {result} fields from hash {key}")
            return result
        except Exception as e:
            logger.error(f"Error deleting hash fields from {key}: {e}")
            return 0

    # ==================== UTILITY OPERATIONS ====================

    def ping(self, **kwargs: Any) -> bool:
        """Test connection to Redis"""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    def flushdb(self, **kwargs: Any) -> bool:
        """Clear all keys in the current database"""
        try:
            result = self.client.flushdb()
            logger.warning(f"Flushed Redis database {self.config.db}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error flushing Redis database: {e}")
            return False

    def close(self):
        """Close Redis connection pool"""
        try:
            self.pool.disconnect()
            logger.info("Closed Redis connection pool")
        except Exception as e:
            logger.error(f"Error closing Redis connection pool: {e}")

