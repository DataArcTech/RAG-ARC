"""Configuration for ChatSessionStorage (Core Layer)"""

from framework.config import AbstractConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.encapsulation.database.cache_db.redis_config import RedisConfig
from typing import Literal, Optional
from core.user_management.chat_session import ChatSessionStorage


class ChatSessionStorageConfig(AbstractConfig):
    """Configuration for ChatSessionStorage - manages chat session metadata in database"""
    type: Literal["chat_session_storage"] = "chat_session_storage"

    # Database configuration
    relational_db_config: PostgreSQLConfig  # Metadata database configuration
    cache_db_config: Optional[RedisConfig] = None  # Redis for session/list cache (optional)
    cache_ttl: Optional[int] = 604800  # Cache TTL in seconds (default: 7 days)

    def build(self) -> ChatSessionStorage:
        return ChatSessionStorage(config=self)

