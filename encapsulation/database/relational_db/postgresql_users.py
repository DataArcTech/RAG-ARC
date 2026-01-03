from typing import Any, Optional, List, Dict, TYPE_CHECKING
import uuid
import logging

from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import select

from ...data_model.orm_models import ChatMessage, ChatSession, User

logger = logging.getLogger(__name__)


class _PostgreSQLUsersMixin:
    def store_user(self, user: User, **kwargs: Any) -> uuid.UUID:
        """Store user metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                session.add(user)
                session.commit()
                logger.debug(f"Stored user: {user.id}")
                return user.id

        except IntegrityError as e:
            logger.error(f"Integrity error storing user (duplicate username?): {e}")
            raise ValueError(f"User with username '{user.user_name}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing user: {e}")
            raise

    def get_user(self, user_id: uuid.UUID, **kwargs: Any) -> Optional[User]:
        """Get user by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                user = session.query(User).filter_by(id=user_id).first()
                if user:
                    # Detach from session to avoid lazy loading issues
                    session.expunge(user)
                return user

        except SQLAlchemyError as e:
            logger.error(f"Database error getting user {user_id}: {e}")
            raise

    def get_user_by_username(self, user_name: str, type: Optional[int] = None, **kwargs: Any) -> Optional[User]:
        """Get user by username (and optional type) using SQLAlchemy ORM."""
        try:
            with self.SessionMaker() as session:
                query = session.query(User).filter_by(user_name=user_name)
                if type is not None:
                    query = query.filter(User.type == type)
                user = query.first()
                if user:
                    # Detach from session to avoid lazy loading issues
                    session.expunge(user)
                return user

        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by username {user_name}: {e}")
            raise

    def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[User]:
        """List all users with pagination using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                query = session.query(User).order_by(User.created_at.desc())

                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                users = query.all()
                # Detach from session
                for user in users:
                    session.expunge(user)

                logger.debug(f"Retrieved {len(users)} users")
                return users

        except SQLAlchemyError as e:
            logger.error(f"Database error listing users: {e}")
            raise

    def update_user(self, user_id: uuid.UUID, updates: dict, **kwargs: Any) -> bool:
        """Update user metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                rows_updated = session.query(User).filter_by(id=user_id).update(updates)
                session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated user: {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False

        except IntegrityError as e:
            logger.error(f"Integrity error updating user (duplicate username?): {e}")
            raise ValueError(f"Username already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating user {user_id}: {e}")
            raise

    def delete_user(self, user_id: uuid.UUID, **kwargs: Any) -> bool:
        """Delete user and dependent chat data using SQLAlchemy ORM.

        Notes:
        - Historical deployments may not have FK constraints with `ON DELETE CASCADE`.
        - We explicitly delete dependent chat sessions and messages to keep behavior consistent.
        """
        try:
            with self.SessionMaker() as session:
                # Delete messages for sessions owned by this user.
                session_ids_select = select(ChatSession.id).where(ChatSession.user_id == user_id)
                session.query(ChatMessage).filter(ChatMessage.session_id.in_(session_ids_select)).delete(
                    synchronize_session=False
                )
                # Delete sessions owned by this user.
                session.query(ChatSession).filter(ChatSession.user_id == user_id).delete(synchronize_session=False)
                # Delete the user row.
                rows_deleted = session.query(User).filter_by(id=user_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted user: {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting user {user_id}: {e}")
            raise

    # ==================== CHAT SESSION MANAGEMENT ====================
