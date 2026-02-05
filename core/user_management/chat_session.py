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

# Default first-page size for list cache (one key per user)
SESSION_LIST_CACHE_LIMIT = 100


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
        """Initialize ChatSessionStorage with metadata store and optional Redis cache"""
        super().__init__(config)
        self.metadata_store = config.relational_db_config.build()
        self.cache_store = config.cache_db_config.build() if getattr(config, "cache_db_config", None) else None
        self.cache_ttl = getattr(config, "cache_ttl", None) or 604800
        if self.cache_store:
            logger.info("ChatSessionStorage initialized with Redis cache")
        else:
            logger.info("ChatSessionStorage initialized without cache (PostgreSQL only)")

    def _session_to_dict(self, session: ChatSession) -> dict:
        """Serialize ChatSession to cache-friendly dict."""
        return {
            "id": str(session.id),
            "user_id": str(session.user_id),
            "name": session.name,
            "is_shared": getattr(session, "is_shared", False),
            "session_metadata": getattr(session, "session_metadata", None),
            "current_task_id": getattr(session, "current_task_id", None),
            "current_task_status": getattr(session, "current_task_status", None),
            "current_task_started_at": (
                session.current_task_started_at.isoformat() if getattr(session, "current_task_started_at", None) else None
            ),
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        }

    def _dict_to_session(self, d: dict) -> Optional[ChatSession]:
        """Deserialize dict from cache to ChatSession."""
        if not d or "id" not in d:
            return None
        created_at = datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now()
        updated_at = datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else datetime.now()
        current_task_started_at = None
        if d.get("current_task_started_at"):
            try:
                current_task_started_at = datetime.fromisoformat(d["current_task_started_at"])
            except (ValueError, TypeError):
                pass
        return ChatSession(
            id=uuid.UUID(d["id"]),
            user_id=uuid.UUID(d["user_id"]),
            name=d.get("name", ""),
            is_shared=d.get("is_shared", False),
            session_metadata=d.get("session_metadata"),
            current_task_id=d.get("current_task_id"),
            current_task_status=d.get("current_task_status"),
            current_task_started_at=current_task_started_at,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _invalidate_user_list_cache(self, user_id: uuid.UUID) -> None:
        """Remove list cache for a user (on create/update/delete)."""
        if not self.cache_store:
            return
        try:
            key = f"chat:session:list:{user_id}"
            self.cache_store.delete(key)
        except Exception as e:
            logger.warning("Failed to invalidate session list cache for user %s: %s", user_id, e)

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

            self._invalidate_user_list_cache(user_id)
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
        Get chat session by ID. Uses Redis cache when configured.
        """
        try:
            cache_key = f"chat:session:meta:{session_id}"
            if self.cache_store:
                try:
                    cached = self.cache_store.get(cache_key)
                    if cached is not None:
                        session = self._dict_to_session(cached)
                        if session is not None:
                            return session
                except Exception as e:
                    logger.warning("Failed to read session from Redis cache: %s", e)
            session = self.metadata_store.get_chat_session(session_id, **kwargs)
            if session and self.cache_store:
                try:
                    self.cache_store.set(cache_key, self._session_to_dict(session), ttl=self.cache_ttl)
                except Exception as e:
                    logger.warning("Failed to write session to Redis cache: %s", e)
            return session
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
        List all chat sessions for a specific user. First page (offset=0) uses Redis when configured.
        """
        try:
            list_key = f"chat:session:list:{user_id}"
            if self.cache_store and offset == 0 and limit > 0 and limit <= SESSION_LIST_CACHE_LIMIT:
                try:
                    cached = self.cache_store.get(list_key)
                    if isinstance(cached, list) and len(cached) >= limit:
                        result = [self._dict_to_session(s) for s in cached[:limit]]
                        result = [x for x in result if x is not None]
                        if len(result) >= limit:
                            return result
                except Exception as e:
                    logger.warning("Failed to read session list from Redis cache: %s", e)
            if self.cache_store and offset == 0 and limit <= SESSION_LIST_CACHE_LIMIT:
                to_fetch = SESSION_LIST_CACHE_LIMIT
                sessions = self.metadata_store.list_chat_sessions_by_user(
                    user_id=user_id, limit=to_fetch, offset=0, **kwargs
                )
                try:
                    self.cache_store.set(
                        list_key,
                        [self._session_to_dict(s) for s in sessions],
                        ttl=self.cache_ttl,
                    )
                except Exception as e:
                    logger.warning("Failed to backfill session list to Redis cache: %s", e)
                return sessions[:limit]
            sessions = self.metadata_store.list_chat_sessions_by_user(
                user_id=user_id,
                limit=limit,
                offset=offset,
                **kwargs
            )
            return sessions
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

            if success and self.cache_store:
                try:
                    self.cache_store.delete(f"chat:session:meta:{session_id}")
                    session = self.metadata_store.get_chat_session(session_id, **kwargs)
                    if session:
                        self._invalidate_user_list_cache(session.user_id)
                except Exception as e:
                    logger.warning("Failed to invalidate session cache: %s", e)
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
            if self.cache_store:
                try:
                    session = self.metadata_store.get_chat_session(session_id, **kwargs)
                    if session:
                        self._invalidate_user_list_cache(session.user_id)
                    self.cache_store.delete(f"chat:session:meta:{session_id}")
                except Exception as e:
                    logger.warning("Failed to invalidate session cache before delete: %s", e)
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

