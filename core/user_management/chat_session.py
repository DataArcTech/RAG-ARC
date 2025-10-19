from typing import (
    Any,
    Optional,
    List,
    TYPE_CHECKING,
)
from datetime import datetime
import logging
import uuid

from encapsulation.data_model.orm_models import ChatSession

from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.core.file_management.storage.chat_session_storage import ChatSessionStorageConfig

logger = logging.getLogger(__name__)


class ChatSessionValidationError(Exception):
    """Raised when chat session validation fails"""
    pass


class StorageOperationError(Exception):
    """Raised when storage operation fails"""
    pass


class ChatSessionStorage(AbstractModule):
    """
    Core chat session storage interface for RAG system.

    Provides high-level chat session management operations including:
    - Session creation and validation
    - Session retrieval and listing
    - Session update and deletion
    - User session isolation

    Architecture:
        Application Layer -> ChatSessionStorage (Core) -> Metadata Storage

    Dependencies:
        metadata_store: RelationalDB implementation (e.g., PostgreSQLDB)
    """

    def __init__(self, config: "ChatSessionStorageConfig"):
        """Initialize ChatSessionStorage with metadata store"""
        super().__init__(config)
        self.metadata_store = config.relational_db_config.build()

    def _validate_session_creation(
        self,
        user_id: uuid.UUID,
        name: str
    ) -> None:
        """Validate session creation parameters"""
        if not user_id:
            raise ChatSessionValidationError("User ID cannot be empty")

        if not name or not name.strip():
            raise ChatSessionValidationError("Session name cannot be empty")

        # Validate session name length
        max_name_length = 255
        if len(name) > max_name_length:
            raise ChatSessionValidationError(f"Session name too long (max {max_name_length} characters)")

    def create_session(
        self,
        user_id: uuid.UUID,
        name: str,
        **kwargs: Any
    ) -> str:
        """
        Create a new chat session.

        Args:
            user_id: User ID who owns this session
            name: Session name
            **kwargs: Additional arguments

        Returns:
            Session ID

        Raises:
            ChatSessionValidationError: If validation fails
            StorageOperationError: If storage operation fails
        """
        try:
            # Validate input
            self._validate_session_creation(user_id, name)

            # Verify user exists
            user = self.metadata_store.get_user(user_id, **kwargs)
            if not user:
                raise ChatSessionValidationError(f"User {user_id} not found")

            # Create session metadata
            session_metadata = ChatSession(
                user_id=user_id,
                name=name,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # Store session metadata
            stored_session_id = self.metadata_store.store_chat_session(session_metadata, **kwargs)

            if not stored_session_id:
                raise StorageOperationError("Failed to store chat session metadata")

            return stored_session_id

        except ChatSessionValidationError:
            raise
        except Exception as e:
            error_msg = f"Chat session creation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StorageOperationError(error_msg)

    def get_session(
        self,
        session_id: uuid.UUID,
        **kwargs: Any
    ) -> Optional[ChatSession]:
        """
        Get chat session by ID.

        Args:
            session_id: Session ID as UUID
            **kwargs: Additional arguments

        Returns:
            ChatSession metadata or None if not found
        """
        try:
            return self.metadata_store.get_chat_session(session_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None

    def list_sessions_by_user(
        self,
        user_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[ChatSession]:
        """
        List all chat sessions for a specific user.

        Args:
            user_id: User ID as UUID
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            **kwargs: Additional arguments

        Returns:
            List of chat session metadata
        """
        try:
            return self.metadata_store.list_chat_sessions_by_user(
                user_id=user_id,
                limit=limit,
                offset=offset,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to list chat sessions for user {user_id}: {e}")
            return []

    def update_session(
        self,
        session_id: uuid.UUID,
        updates: dict,
        **kwargs: Any
    ) -> bool:
        """
        Update chat session metadata.

        Args:
            session_id: Session ID as UUID
            updates: Dictionary of fields to update
            **kwargs: Additional arguments

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # Add updated_at timestamp
            updates['updated_at'] = datetime.now()

            logger.info(f"Updating chat session {session_id}: {list(updates.keys())}")
            success = self.metadata_store.update_chat_session(session_id, updates, **kwargs)

            if success:
                logger.info(f"Successfully updated chat session {session_id}")
            else:
                logger.warning(f"Failed to update chat session {session_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update chat session {session_id}: {e}")
            return False

    def delete_session(
        self,
        session_id: uuid.UUID,
        **kwargs: Any
    ) -> bool:
        """
        Delete chat session.

        Note: This will cascade delete all associated messages
        due to foreign key constraints.

        Args:
            session_id: Session ID as UUID
            **kwargs: Additional arguments

        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            logger.info(f"Deleting chat session {session_id}")
            success = self.metadata_store.delete_chat_session(session_id, **kwargs)

            if success:
                logger.info(f"Successfully deleted chat session {session_id}")
            else:
                logger.warning(f"Failed to delete chat session {session_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete chat session {session_id}: {e}")
            return False

    def get_user_session_count(
        self,
        user_id: uuid.UUID
    ) -> int:
        """
        Get the number of sessions for a user.
        """
        try:
            return self.metadata_store.get_user_session_count(user_id)
        except Exception as e:
            logger.error(f"Failed to get user session count for user {user_id}: {e}")
            return 0

    def verify_session_ownership(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        **kwargs: Any
    ) -> bool:
        """
        Verify that a session belongs to a specific user.

        Args:
            session_id: Session ID as UUID
            user_id: User ID as UUID
            **kwargs: Additional arguments

        Returns:
            True if session belongs to user, False otherwise
        """
        try:
            session = self.get_session(session_id, **kwargs)
            if not session:
                return False
            return session.user_id == user_id
        except Exception as e:
            logger.error(f"Failed to verify session ownership: {e}")
            return False

