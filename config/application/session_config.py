from framework.config import AbstractConfig
from config.core.file_management.storage.chat_session_storage import ChatSessionStorageConfig
from typing import Literal
from application.account.chat_session import ChatSessionManager


class ChatSessionConfig(AbstractConfig):
    """
    Application-level configuration for ChatSessionManager.
    Provides dependencies for running the session management module.
    """
    type: Literal["chat_session"] = "chat_session"

    # Storage backend configuration (core layer)
    session_storage_config: ChatSessionStorageConfig

    def build(self) -> ChatSessionManager:
        return ChatSessionManager(config=self)
