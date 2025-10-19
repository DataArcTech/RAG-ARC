from framework.config import AbstractConfig
from config.core.file_management.storage.chat_message_storage import ChatMessageStorageConfig
from typing import Literal
from application.account.chat_message import ChatMessageManager


class ChatMessageManagerConfig(AbstractConfig):
    """
    Application-level configuration for ChatMessageManager.
    Provides dependencies for running the message management module.
    """
    type: Literal["chat_message"] = "chat_message"

    # Storage backend configuration (core layer)
    message_storage_config: ChatMessageStorageConfig

    def build(self) -> ChatMessageManager:
        return ChatMessageManager(config=self)
