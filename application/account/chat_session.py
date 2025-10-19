from typing import Any, Optional, List, TYPE_CHECKING
from datetime import datetime
import logging
import uuid

from encapsulation.data_model.orm_models import ChatSession
from framework.module import AbstractModule
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.application.session_config import ChatSessionConfig

logger = logging.getLogger(__name__)


class ChatSessionManager(AbstractModule):
    """
    Application layer chat session manager for RAG system.

    Provides high-level chat session management operations by delegating to
    the core ChatSessionStorage implementation.

    Architecture:
        Application Layer -> ChatSessionManager -> ChatSessionStorage (Core) -> PostgreSQL

    Dependencies:
        session_storage: ChatSessionStorage implementation
    """

    def __init__(self, config: "ChatSessionConfig"):
        """Initialize ChatSessionManager with session storage"""
        super().__init__(config)
        self.session_storage = config.session_storage_config.build()

    def create_session(
        self,
        user_id: uuid.UUID,
        session_name: str = "New Chat"
    ) -> str:
        """
        Create a new chat session.

        Args:
            user_id: User ID who owns the session
            session_name: Name for the session

        Returns:
            Session ID
        """
        return self.session_storage.create_session(user_id, session_name)

    def get_session(
        self,
        session_id: uuid.UUID,
    ) -> Optional[ChatSession]:
        """
        Get session by ID.

        Args:
            session_id: Session ID as UUID

        Returns:
            Session metadata or None if not found
        """
        return self.session_storage.get_session(session_id)

    def list_sessions_by_user(
        self,
        user_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ChatSession]:
        """
        List sessions for a user with pagination.

        Args:
            user_id: User ID as UUID
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of session metadata
        """
        return self.session_storage.list_sessions_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    def update_session(
        self,
        session_id: uuid.UUID,
        updates: dict,
    ) -> bool:
        """
        Update session metadata.

        Args:
            session_id: Session ID as UUID
            updates: Dictionary of fields to update

        Returns:
            True if update succeeded, False otherwise
        """
        # Add updated_at timestamp
        updates['updated_at'] = datetime.now()
        
        # Delegate to core storage
        return self.session_storage.update_session(session_id, updates)

    def delete_session(
        self,
        session_id: uuid.UUID
    ) -> bool:
        """
        Delete session.

        Note: This will cascade delete all associated messages
        due to foreign key constraints.

        Args:
            session_id: Session ID as UUID

        Returns:
            True if deletion succeeded, False otherwise
        """
        return self.session_storage.delete_session(session_id)

    def get_user_session_count(
        self,
        user_id: uuid.UUID
    ) -> int:
        """
        Get the number of sessions for a user.

        Args:
            user_id: User ID as UUID
            **kwargs: Additional arguments

        Returns:
            Number of sessions
        """
        return self.session_storage.get_user_session_count(user_id)
