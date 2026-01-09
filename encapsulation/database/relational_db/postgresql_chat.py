from typing import Any, Optional, List
import uuid
import logging

from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ...data_model.orm_models import (
    ChatSession,
    ChatMessage,
)

logger = logging.getLogger(__name__)


class _PostgreSQLChatMixin:
    def store_chat_session(self, session: ChatSession, **kwargs: Any) -> str:
        """Store chat session metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                db_session.add(session)
                db_session.commit()
                logger.debug(f"Stored chat session: {session.id}")
                return str(session.id)

        except IntegrityError as e:
            logger.error(f"Integrity error storing chat session (invalid user_id?): {e}")
            raise ValueError(f"Invalid user_id or constraint violation")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing chat session: {e}")
            raise

    def get_chat_session(self, session_id: uuid.UUID, **kwargs: Any) -> Optional[ChatSession]:
        """Get chat session by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                chat_session = db_session.query(ChatSession).filter_by(id=session_id).first()
                if chat_session:
                    # Detach from session to avoid lazy loading issues
                    db_session.expunge(chat_session)
                return chat_session

        except SQLAlchemyError as e:
            logger.error(f"Database error getting chat session {session_id}: {e}")
            raise

    def list_chat_sessions_by_user(
        self,
        user_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[ChatSession]:
        """List all chat sessions for a specific user using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                query = db_session.query(ChatSession).filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc())

                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                sessions = query.all()
                # Detach from session
                for s in sessions:
                    db_session.expunge(s)

                logger.debug(f"Retrieved {len(sessions)} chat sessions for user {user_id}")
                return sessions

        except SQLAlchemyError as e:
            logger.error(f"Database error listing chat sessions for user {user_id}: {e}")
            raise

    def get_user_session_count(self, user_id: uuid.UUID, **kwargs: Any) -> int:
        """Get the number of sessions for a user using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                return db_session.query(ChatSession).filter_by(user_id=user_id).count()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user session count for user {user_id}: {e}")
            return 0

    def update_chat_session(self, session_id: uuid.UUID, updates: dict, **kwargs: Any) -> bool:
        """Update chat session metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                rows_updated = db_session.query(ChatSession).filter_by(id=session_id).update(updates)
                db_session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated chat session: {session_id}")
                    return True
                else:
                    logger.warning(f"No chat session found with ID: {session_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating chat session {session_id}: {e}")
            raise

    def delete_chat_session(self, session_id: uuid.UUID, **kwargs: Any) -> bool:
        """Delete chat session using SQLAlchemy ORM (cascades to messages)"""
        try:
            with self.SessionMaker() as db_session:
                # First, delete all messages for this session to avoid foreign key constraint violation
                messages_deleted = db_session.query(ChatMessage).filter_by(session_id=session_id).delete(synchronize_session=False)
                if messages_deleted > 0:
                    logger.info(f"Deleted {messages_deleted} messages for session {session_id}")
                
                # Then delete the session
                rows_deleted = db_session.query(ChatSession).filter_by(id=session_id).delete()
                db_session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted chat session: {session_id}")
                    return True
                else:
                    logger.warning(f"No chat session found with ID: {session_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chat session {session_id}: {e}")
            raise

    # ==================== CHAT MESSAGE MANAGEMENT ====================

    def _clean_null_chars(self, obj: Any) -> Any:
        """Recursively remove NULL characters (\u0000) from JSON-serializable objects"""
        if isinstance(obj, str):
            return obj.replace('\x00', '').replace('\u0000', '')
        elif isinstance(obj, dict):
            return {k: self._clean_null_chars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_null_chars(item) for item in obj]
        else:
            return obj

    def store_chat_message(self, message: ChatMessage, **kwargs: Any) -> Optional[ChatMessage]:
        """Store chat message metadata using SQLAlchemy ORM"""
        try:
            # Clean NULL characters from all JSON fields before storing
            if message.content:
                message.content = self._clean_null_chars(message.content)
            if message.sources:
                message.sources = self._clean_null_chars(message.sources)
            if message.source_file_ids:
                message.source_file_ids = self._clean_null_chars(message.source_file_ids)
            if message.subgraph_data:
                message.subgraph_data = self._clean_null_chars(message.subgraph_data)
            if message.raw_llm_response:
                message.raw_llm_response = self._clean_null_chars(message.raw_llm_response)
            if message.raw_mindmap_response:
                message.raw_mindmap_response = self._clean_null_chars(message.raw_mindmap_response)
            
            with self.SessionMaker() as db_session:
                db_session.add(message)
                db_session.commit()
                # Expunge the object to avoid lazy loading issues when accessing attributes
                # This prevents SQLAlchemy from trying to reload the object from the database
                # which could fail if the database schema doesn't match the model exactly
                message_id = message.id
                db_session.expunge(message)
                logger.debug(f"Stored chat message: {message_id}")
                return message

        except IntegrityError as e:
            logger.error(f"Integrity error storing chat message (invalid session_id?): {e}")
            raise ValueError(f"Invalid session_id or constraint violation")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing chat message: {e}")
            raise

    def get_chat_message(self, message_id: uuid.UUID, **kwargs: Any) -> Optional[ChatMessage]:
        """Get chat message by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                message = db_session.query(ChatMessage).filter_by(id=message_id).first()
                if message:
                    # Detach from session to avoid lazy loading issues
                    db_session.expunge(message)
                return message

        except SQLAlchemyError as e:
            logger.error(f"Database error getting chat message {message_id}: {e}")
            raise

    def list_chat_messages_by_session(
        self,
        session_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[ChatMessage]:
        """List all chat messages for a specific session using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                query = db_session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc())

                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                messages = query.all()
                # Detach from session
                for msg in messages:
                    db_session.expunge(msg)

                logger.debug(f"Retrieved {len(messages)} chat messages for session {session_id}")
                return messages

        except SQLAlchemyError as e:
            logger.error(f"Database error listing chat messages for session {session_id}: {e}")
            raise

    def delete_chat_message(self, message_id: uuid.UUID, **kwargs: Any) -> bool:
        """Delete chat message using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                rows_deleted = db_session.query(ChatMessage).filter_by(id=message_id).delete()
                db_session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted chat message: {message_id}")
                    return True
                else:
                    logger.warning(f"No chat message found with ID: {message_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chat message {message_id}: {e}")
            raise

    def delete_chat_messages_by_session(self, session_id: uuid.UUID, **kwargs: Any) -> int:
        """Delete all chat messages for a specific session using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                rows_deleted = db_session.query(ChatMessage).filter_by(session_id=session_id).delete()
                db_session.commit()

                logger.info(f"Deleted {rows_deleted} chat messages for session {session_id}")
                return rows_deleted

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chat messages for session {session_id}: {e}")
            raise

    # ==================== FILE PERMISSION MANAGEMENT ====================


