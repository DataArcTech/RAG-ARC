"""Configuration for ChatSessionStorage (Core Layer)"""

from framework.config import AbstractConfig
from core.file_management.storage.chat_session import ChatSessionStorage
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from typing import Literal


class ChatSessionStorageConfig(AbstractConfig):
    """Configuration for ChatSessionStorage - manages chat session metadata in database"""
    type: Literal["chat_session_storage"] = "chat_session_storage"

    # Database configuration
    relational_db_config: PostgreSQLConfig  # Metadata database configuration

    def build(self) -> ChatSessionStorage:
        return ChatSessionStorage(self)

