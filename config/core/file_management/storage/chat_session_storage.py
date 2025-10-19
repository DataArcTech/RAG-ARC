"""Configuration for ChatSessionStorage (Core Layer)"""

from framework.config import AbstractConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from typing import Literal
from core.user_management.chat_session import ChatSessionStorage

class ChatSessionStorageConfig(AbstractConfig):
    """Configuration for ChatSessionStorage - manages chat session metadata in database"""
    type: Literal["chat_session_storage"] = "chat_session_storage"

    # Database configuration
    relational_db_config: PostgreSQLConfig  # Metadata database configuration

    def build(self) -> ChatSessionStorage:
        return ChatSessionStorage(config=self)

