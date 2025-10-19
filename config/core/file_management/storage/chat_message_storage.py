"""Configuration for ChatMessageStorage (Core Layer)"""

from framework.config import AbstractConfig
from core.user_management.chat_message import ChatMessageStorage
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.encapsulation.database.cache_db.redis_config import RedisConfig
from typing import Literal, Optional


class ChatMessageStorageConfig(AbstractConfig):
    """
    Configuration for ChatMessageStorage - manages chat message metadata with dual-layer storage.

    Dual-layer storage architecture:
    - Redis (cache_db_config): Hot data cache for recent messages (fast read/write)
    - PostgreSQL (relational_db_config): Cold data storage for all messages (persistent)

    Read flow:
    1. Try to read from Redis first (fast)
    2. If not found or insufficient data, read from PostgreSQL
    3. Backfill Redis with PostgreSQL data

    Write flow:
    1. Write to Redis immediately (fast response)
    2. Asynchronously write to PostgreSQL (persistent)
    """
    type: Literal["chat_message_storage"] = "chat_message_storage"

    # Database configurations
    relational_db_config: PostgreSQLConfig  # PostgreSQL for persistent storage
    cache_db_config: Optional[RedisConfig] = None  # Redis for hot data cache (optional)

    # Cache configuration
    cache_max_messages: int = 100  # Maximum messages to keep in Redis per session
    cache_ttl: Optional[int] = 604800  # Cache TTL in seconds (default: 7 days, None = no expiration)

    def build(self) -> ChatMessageStorage:
        return ChatMessageStorage(config=self)

