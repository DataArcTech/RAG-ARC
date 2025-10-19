from typing import (
    Any,
    Optional,
    List,
    Dict,
    TYPE_CHECKING,
)
import logging
import uuid

from encapsulation.data_model.orm_models import ChatMessage

from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.core.file_management.storage.chat_message_storage import ChatMessageStorageConfig
    from encapsulation.database.cache_db.redis_db import RedisDB

logger = logging.getLogger(__name__)


class ChatMessageValidationError(Exception):
    """Raised when chat message validation fails"""
    pass


class StorageOperationError(Exception):
    """Raised when storage operation fails"""
    pass


class ChatMessageStorage(AbstractModule):
    """
    Core chat message storage interface for RAG system with dual-layer storage.

    Provides high-level chat message management operations including:
    - Message creation and validation
    - Message retrieval and listing
    - Message deletion
    - Session message isolation

    Dual-layer storage architecture:
    - Redis (cache_store): Hot data cache for recent messages (fast read/write)
    - PostgreSQL (metadata_store): Cold data storage for all messages (persistent)

    Message content structure:
        {
            "role": "user" | "assistant" | "system",
            "content": "message text",
            "metadata": {
                "model": "...",
                "tokens": {...},
                "sources": [...],
                ...
            }
        }

    Architecture:
        Application Layer -> ChatMessageStorage (Core) -> Redis + PostgreSQL

    Dependencies:
        metadata_store: RelationalDB implementation (e.g., PostgreSQLDB)
        cache_store: CacheDB implementation (e.g., RedisDB) - optional
    """

    def __init__(self, config: "ChatMessageStorageConfig"):
        """Initialize ChatMessageStorage with metadata store and optional cache store"""
        super().__init__(config)
        self.metadata_store = config.relational_db_config.build()

        # Initialize Redis cache if configured
        self.cache_store: Optional["RedisDB"] = None
        if config.cache_db_config:
            self.cache_store = config.cache_db_config.build()
            logger.info("ChatMessageStorage initialized with Redis cache")
        else:
            logger.info("ChatMessageStorage initialized without cache (PostgreSQL only)")

        # Cache configuration
        self.cache_max_messages = config.cache_max_messages
        self.cache_ttl = config.cache_ttl

    def _get_cache_key(self, session_id: str) -> str:
        """Get Redis cache key for a session's messages"""
        return f"chat:session:{session_id}:messages"

    def _message_to_cache_format(self, message: ChatMessage) -> Dict[str, Any]:
        """Convert ChatMessage ORM object to cache-friendly dict"""
        return {
            "message_id": str(message.id),
            "session_id": str(message.session_id),
            "role": message.content.get("role", "user"),
            "content": message.content.get("content", ""),
            "metadata": message.content.get("metadata", {}),
            "created_at": message.created_at.isoformat() if message.created_at else None
        }

    def _cache_format_to_message_dict(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert cache format to message dict for API response"""
        return {
            "message_id": cache_data.get("message_id"),
            "session_id": cache_data.get("session_id"),
            "role": cache_data.get("role", "user"),
            "content": cache_data.get("content", ""),
            "metadata": cache_data.get("metadata", {}),
            "created_at": cache_data.get("created_at")
        }

    def _validate_message_creation(
        self,
        chat_message: ChatMessage
    ) -> None:
        """Validate message creation parameters"""
        if not chat_message.session_id:
            raise ChatMessageValidationError("Session ID cannot be empty")

        if not chat_message.content:
            raise ChatMessageValidationError("Message content cannot be empty")

        # Validate content structure
        if not isinstance(chat_message.content, dict):
            raise ChatMessageValidationError("Message content must be a dictionary")

        # Validate required fields
        if 'role' not in chat_message.content:
            raise ChatMessageValidationError("Message content must include 'role' field")

        if 'content' not in chat_message.content:
            raise ChatMessageValidationError("Message content must include 'content' field")

        # Validate role
        valid_roles = ['user', 'assistant', 'system']
        if chat_message.content['role'] not in valid_roles:
            raise ChatMessageValidationError(f"Invalid role: {chat_message.content['role']}. Must be one of {valid_roles}")

        # Validate content text
        if not isinstance(chat_message.content['content'], str):
            raise ChatMessageValidationError("Message 'content' field must be a string")

        if not chat_message.content['content'].strip():
            raise ChatMessageValidationError("Message content text cannot be empty")

    def create_message(
        self,
        chat_message: ChatMessage,
        **kwargs: Any
    ) -> ChatMessage:
        """
        Create a new chat message with dual-layer storage.

        Write flow:
        1. Write to Redis immediately (fast response)
        2. Write to PostgreSQL (persistent storage)
        3. Trim Redis list to keep only recent messages

        Args:
            chat_message: ChatMessage object, 
            **kwargs: Additional arguments

        Returns:
            ChatMessage object

        Raises:
            ChatMessageValidationError: If validation fails
            StorageOperationError: If storage operation fails
        """
        try:
            # Validate input
            self._validate_message_creation(chat_message)

            # Verify session exists
            session = self.metadata_store.get_chat_session(chat_message.session_id, **kwargs)
            if not session:
                raise ChatMessageValidationError(f"Chat session {chat_message.session_id} not found")

            # Write to PostgreSQL (persistent storage)
            logger.info(f"Creating chat message for session {chat_message.session_id} (role: {chat_message.content.get('role', 'user')})")
            chat_message = self.metadata_store.store_chat_message(chat_message, **kwargs)

            if not chat_message:
                raise StorageOperationError("Failed to create chat message")

            logger.info(f"Successfully created chat message (message_id: {chat_message.id})")
            return chat_message

        except ChatMessageValidationError:
            raise
        except Exception as e:
            error_msg = f"Chat message creation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StorageOperationError(error_msg)

    def get_message(
        self,
        message_id: uuid.UUID
    ) -> Optional[ChatMessage]:
        """
        Get chat message by ID.

        Args:
            message_id: Message ID as UUID
            **kwargs: Additional arguments

        Returns:
            ChatMessage metadata or None if not found
        """
        try:
            return self.metadata_store.get_chat_message(message_id)
        except Exception as e:
            logger.error(f"Failed to get chat message {message_id}: {e}")
            return None

    def list_messages_by_session(
        self,
        session_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[ChatMessage]:
        """
        List all messages for a specific session with dual-layer read.

        Read flow:
        1. Try to read from Redis first (fast)
        2. If not found or insufficient data, read from PostgreSQL
        3. Backfill Redis with PostgreSQL data

        Args:
            session_id: Session ID as UUID
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            **kwargs: Additional arguments

        Returns:
            List of chat message metadata, ordered by created_at (oldest first)
        """
        try:
            # Read from PostgreSQL
            messages = self.metadata_store.list_chat_messages_by_session(
                session_id=session_id,
                limit=limit,
                offset=offset,
                **kwargs
            )

            return messages

        except Exception as e:
            logger.error(f"Failed to list chat messages for session {session_id}: {e}")
            return []

    def delete_message(
        self,
        message_id: uuid.UUID,
        **kwargs: Any
    ) -> bool:
        """
        Delete chat message from both Redis and PostgreSQL.

        Args:
            message_id: Message ID as UUID
            **kwargs: Additional arguments

        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            logger.info(f"Deleting chat message {message_id}")

            # Get message to find session_id for cache invalidation
            message = self.metadata_store.get_chat_message(message_id, **kwargs)

            # Delete from PostgreSQL
            success = self.metadata_store.delete_chat_message(message_id, **kwargs)

            if success:
                # Invalidate Redis cache for this session
                if message and self.cache_store:
                    try:
                        cache_key = self._get_cache_key(str(message.session_id))
                        self.cache_store.delete(cache_key)
                        logger.debug(f"Invalidated Redis cache for session {message.session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to invalidate Redis cache: {e}")

                logger.info(f"Successfully deleted chat message {message_id}")
            else:
                logger.warning(f"Failed to delete chat message {message_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete chat message {message_id}: {e}")
            return False

    def delete_messages_by_session(
        self,
        session_id: uuid.UUID,
        **kwargs: Any
    ) -> int:
        """
        Delete all messages for a specific session from both Redis and PostgreSQL.

        Args:
            session_id: Session ID as UUID
            **kwargs: Additional arguments

        Returns:
            Number of messages deleted
        """
        try:
            logger.info(f"Deleting all messages for session {session_id}")

            # Delete from PostgreSQL
            count = self.metadata_store.delete_chat_messages_by_session(session_id, **kwargs)

            # Delete from Redis
            if self.cache_store:
                try:
                    cache_key = self._get_cache_key(session_id)
                    self.cache_store.delete(cache_key)
                    logger.debug(f"Deleted Redis cache for session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete Redis cache: {e}")

            logger.info(f"Successfully deleted {count} messages for session {session_id}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete messages for session {session_id}: {e}")
            return 0
