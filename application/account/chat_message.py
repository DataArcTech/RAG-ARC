from typing import Any, Optional, List, Dict, TYPE_CHECKING
from datetime import datetime
import logging
import uuid

from encapsulation.data_model.orm_models import ChatMessage
from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.application.chat_message_config import ChatMessageManagerConfig

logger = logging.getLogger(__name__)


class ChatMessageManager(AbstractModule):
    """
    Application layer chat message manager for RAG system.

    Provides high-level chat message management operations by delegating to
    the core ChatMessageStorage implementation.

    Architecture:
        Application Layer -> ChatMessageManager -> ChatMessageStorage (Core) -> Redis + PostgreSQL

    Dependencies:
        message_storage: ChatMessageStorage implementation
    """

    def __init__(self, config: "ChatMessageManagerConfig"):
        """Initialize ChatMessageManager with message storage"""
        super().__init__(config)
        self.message_storage = config.message_storage_config.build()

    def create_message(
        self,
        chat_message: ChatMessage
    ) -> ChatMessage:
        """
        Create a new chat message.

        Args:
            chat_message: ChatMessage object

        Returns:
            ChatMessage object
        """
        return self.message_storage.create_message(chat_message)

    def get_message(
        self,
        message_id: uuid.UUID,
    ) -> Optional[ChatMessage]:
        """
        Get message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message metadata or None if not found
        """
        return self.message_storage.get_message(message_id)

    def list_messages_by_session(
        self,
        session_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ChatMessage]:
        """
        List messages for a session with pagination.

        Args:
            session_id: Session ID as UUID
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List of message metadata, ordered by created_at (oldest first)
        """
        return self.message_storage.list_messages_by_session(
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

    def delete_message(
        self,
        message_id: uuid.UUID
    ) -> bool:
        """
        Delete message.

        Args:
            message_id: Message ID as UUID

        Returns:
            True if deletion succeeded, False otherwise
        """
        return self.message_storage.delete_message(message_id)

    def delete_messages_by_session(
        self,
        session_id: uuid.UUID
    ) -> int:
        """
        Delete all messages for a session.

        Args:
            session_id: Session ID as UUID

        Returns:
            Number of messages deleted
        """
        return self.message_storage.delete_messages_by_session(session_id)

    def get_session_message_count(
        self,
        session_id: uuid.UUID
    ) -> int:
        """
        Get the number of messages for a session.

        Args:
            session_id: Session ID as UUID

        Returns:
            Number of messages
        """
        # Get all messages and count them
        messages = self.list_messages_by_session(session_id, limit=10000)
        return len(messages)