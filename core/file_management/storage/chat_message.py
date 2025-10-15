from typing import (
    Any,
    Optional,
    List,
    Dict,
    TYPE_CHECKING,
)
from datetime import datetime
import logging
import uuid
import asyncio
import json

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

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return str(uuid.uuid4())

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
        session_id: str,
        content: Dict[str, Any]
    ) -> None:
        """Validate message creation parameters"""
        if not session_id or not session_id.strip():
            raise ChatMessageValidationError("Session ID cannot be empty")

        if not content:
            raise ChatMessageValidationError("Message content cannot be empty")

        # Validate session_id format (should be valid UUID)
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ChatMessageValidationError("Invalid session ID format")

        # Validate content structure
        if not isinstance(content, dict):
            raise ChatMessageValidationError("Message content must be a dictionary")

        # Validate required fields
        if 'role' not in content:
            raise ChatMessageValidationError("Message content must include 'role' field")

        if 'content' not in content:
            raise ChatMessageValidationError("Message content must include 'content' field")

        # Validate role
        valid_roles = ['user', 'assistant', 'system']
        if content['role'] not in valid_roles:
            raise ChatMessageValidationError(f"Invalid role: {content['role']}. Must be one of {valid_roles}")

        # Validate content text
        if not isinstance(content['content'], str):
            raise ChatMessageValidationError("Message 'content' field must be a string")

        if not content['content'].strip():
            raise ChatMessageValidationError("Message content text cannot be empty")

    def create_message(
        self,
        session_id: str,
        content: Dict[str, Any],
        **kwargs: Any
    ) -> str:
        """
        Create a new chat message with dual-layer storage.

        Write flow:
        1. Write to Redis immediately (fast response)
        2. Write to PostgreSQL (persistent storage)
        3. Trim Redis list to keep only recent messages

        Args:
            session_id: Session ID this message belongs to
            content: Message content dictionary with structure:
                {
                    "role": "user" | "assistant" | "system",
                    "content": "message text",
                    "metadata": {...}  # optional
                }
            **kwargs: Additional arguments

        Returns:
            Message ID

        Raises:
            ChatMessageValidationError: If validation fails
            StorageOperationError: If storage operation fails
        """
        try:
            # Validate input
            self._validate_message_creation(session_id, content)

            # Verify session exists
            session = self.metadata_store.get_chat_session(session_id, **kwargs)
            if not session:
                raise ChatMessageValidationError(f"Chat session {session_id} not found")

            # Generate message ID
            message_id = self._generate_message_id()

            # Create message metadata
            message_metadata = ChatMessage(
                id=uuid.UUID(message_id),
                session_id=uuid.UUID(session_id),
                content=content,
                created_at=datetime.now()
            )

            # 1. Write to Redis first (fast response)
            if self.cache_store:
                try:
                    cache_key = self._get_cache_key(session_id)
                    cache_data = self._message_to_cache_format(message_metadata)

                    # Push to list (newest at head)
                    self.cache_store.lpush(cache_key, cache_data)

                    # Trim to keep only recent messages
                    self.cache_store.ltrim(cache_key, 0, self.cache_max_messages - 1)

                    # Set TTL if configured
                    if self.cache_ttl:
                        self.cache_store.expire(cache_key, self.cache_ttl)

                    logger.debug(f"Cached message {message_id} in Redis")
                except Exception as e:
                    logger.warning(f"Failed to cache message in Redis: {e}")
                    # Continue even if Redis fails

            # 2. Write to PostgreSQL (persistent storage)
            logger.info(f"Creating chat message for session {session_id} (message_id: {message_id}, role: {content['role']})")
            stored_message_id = self.metadata_store.store_chat_message(message_metadata, **kwargs)

            if not stored_message_id:
                raise StorageOperationError("Failed to store chat message metadata")

            logger.info(f"Successfully created chat message (message_id: {message_id})")
            return message_id

        except ChatMessageValidationError:
            raise
        except Exception as e:
            error_msg = f"Chat message creation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StorageOperationError(error_msg)

    def get_message(
        self,
        message_id: str,
        **kwargs: Any
    ) -> Optional[ChatMessage]:
        """
        Get chat message by ID.

        Args:
            message_id: Message ID
            **kwargs: Additional arguments

        Returns:
            ChatMessage metadata or None if not found
        """
        try:
            return self.metadata_store.get_chat_message(message_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get chat message {message_id}: {e}")
            return None

    def list_messages_by_session(
        self,
        session_id: str,
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
            session_id: Session ID
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            **kwargs: Additional arguments

        Returns:
            List of chat message metadata, ordered by created_at (oldest first)
        """
        try:
            # If offset > 0, skip Redis and go directly to PostgreSQL
            # (Redis only stores recent messages)
            if offset == 0 and self.cache_store:
                try:
                    cache_key = self._get_cache_key(session_id)
                    cached_messages = self.cache_store.lrange(cache_key, 0, limit - 1)

                    if cached_messages and len(cached_messages) >= limit:
                        # Redis has enough data, convert to ChatMessage objects
                        logger.debug(f"Retrieved {len(cached_messages)} messages from Redis cache")

                        # Convert cache format to ChatMessage objects
                        # Note: Redis stores newest first, but we need oldest first
                        messages = []
                        for cache_data in reversed(cached_messages):
                            msg_dict = cache_data
                            # Create a minimal ChatMessage object
                            msg = ChatMessage(
                                id=uuid.UUID(msg_dict["message_id"]),
                                session_id=uuid.UUID(msg_dict["session_id"]),
                                content={
                                    "role": msg_dict["role"],
                                    "content": msg_dict["content"],
                                    "metadata": msg_dict.get("metadata", {})
                                },
                                created_at=datetime.fromisoformat(msg_dict["created_at"]) if msg_dict.get("created_at") else datetime.now()
                            )
                            messages.append(msg)

                        return messages

                except Exception as e:
                    logger.warning(f"Failed to read from Redis cache: {e}")
                    # Fall through to PostgreSQL

            # Read from PostgreSQL
            messages = self.metadata_store.list_chat_messages_by_session(
                session_id=session_id,
                limit=limit,
                offset=offset,
                **kwargs
            )

            # Backfill Redis if offset == 0 and cache is enabled
            if offset == 0 and self.cache_store and messages:
                try:
                    cache_key = self._get_cache_key(session_id)

                    # Clear existing cache
                    self.cache_store.delete(cache_key)

                    # Push messages to Redis (newest first)
                    for msg in reversed(messages[:self.cache_max_messages]):
                        cache_data = self._message_to_cache_format(msg)
                        self.cache_store.lpush(cache_key, cache_data)

                    # Set TTL if configured
                    if self.cache_ttl:
                        self.cache_store.expire(cache_key, self.cache_ttl)

                    logger.debug(f"Backfilled Redis cache with {len(messages)} messages")
                except Exception as e:
                    logger.warning(f"Failed to backfill Redis cache: {e}")

            return messages

        except Exception as e:
            logger.error(f"Failed to list chat messages for session {session_id}: {e}")
            return []

    def delete_message(
        self,
        message_id: str,
        **kwargs: Any
    ) -> bool:
        """
        Delete chat message from both Redis and PostgreSQL.

        Args:
            message_id: Message ID
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
        session_id: str,
        **kwargs: Any
    ) -> int:
        """
        Delete all messages for a specific session from both Redis and PostgreSQL.

        Args:
            session_id: Session ID
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

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session in a format suitable for LLM APIs.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return (most recent)
            **kwargs: Additional arguments

        Returns:
            List of message dictionaries in format:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        try:
            messages = self.list_messages_by_session(
                session_id=session_id,
                limit=limit,
                offset=0,
                **kwargs
            )

            # Convert to LLM API format
            history = []
            for msg in messages:
                if msg.content and isinstance(msg.content, dict):
                    history.append({
                        "role": msg.content.get("role", "user"),
                        "content": msg.content.get("content", "")
                    })

            return history

        except Exception as e:
            logger.error(f"Failed to get conversation history for session {session_id}: {e}")
            return []

