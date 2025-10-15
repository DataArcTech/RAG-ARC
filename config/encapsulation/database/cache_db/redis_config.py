"""Configuration for Redis cache database"""

from framework.config import AbstractConfig
from encapsulation.database.cache_db.redis_db import RedisDB
from typing import Literal, Optional


class RedisConfig(AbstractConfig):
    """
    Configuration for Redis cache database.
    
    Redis is used for:
    - Hot data caching (e.g., recent chat messages)
    - Session management
    - Rate limiting
    - Temporary data storage
    
    Persistence:
    - RDB (Redis Database): Point-in-time snapshots
    - AOF (Append Only File): Log of all write operations
    - Configure on Redis server side (redis.conf)
    
    Example redis.conf for persistence:
        # RDB: Save snapshot every 60 seconds if at least 1 key changed
        save 60 1
        
        # AOF: Enable append-only file
        appendonly yes
        appendfsync everysec
    """
    
    # Discriminator for config type identification
    type: Literal["redis"] = "redis"

    # Redis connection configuration
    host: str = "localhost"  # Redis server host
    port: int = 6379  # Redis server port
    db: int = 0  # Redis database number (0-15)
    password: Optional[str] = None  # Redis password (optional)
    
    # Connection pool configuration
    max_connections: int = 50  # Maximum connections in pool
    decode_responses: bool = True  # Decode responses to strings

    def build(self) -> RedisDB:
        return RedisDB(self)

