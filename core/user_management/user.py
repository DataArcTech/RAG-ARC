from typing import (
    Any,
    Optional,
    List,
    TYPE_CHECKING,
)
from datetime import datetime
import logging
import uuid

from encapsulation.data_model.orm_models import User

from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.core.file_management.storage.user_storage import UserStorageConfig

logger = logging.getLogger(__name__)


class UserValidationError(Exception):
    """Raised when user validation fails"""
    pass


class StorageOperationError(Exception):
    """Raised when storage operation fails"""
    pass


class UserStorage(AbstractModule):
    """
    Core user storage interface for RAG system.

    Provides high-level user management operations including:
    - User creation and validation
    - User authentication support
    - User retrieval and listing
    - User update and deletion

    Architecture:
        Application Layer -> UserStorage (Core) -> Metadata Storage

    Dependencies:
        metadata_store: RelationalDB implementation (e.g., PostgreSQLDB)
    """

    def __init__(self, config: "UserStorageConfig"):
        """Initialize UserStorage with metadata store"""
        super().__init__(config)
        self.metadata_store = config.relational_db_config.build()

    def _validate_user_creation(
        self,
        user_name: str,
        hashed_password: str
    ) -> None:
        """Validate user creation parameters"""
        if not user_name or not user_name.strip():
            raise UserValidationError("Username cannot be empty")

        if not hashed_password or not hashed_password.strip():
            raise UserValidationError("Password cannot be empty")

        # Validate username length
        min_username_length = 3
        max_username_length = 255
        if len(user_name) < min_username_length:
            raise UserValidationError(f"Username too short (min {min_username_length} characters)")
        if len(user_name) > max_username_length:
            raise UserValidationError(f"Username too long (max {max_username_length} characters)")

        # Validate username format (alphanumeric, underscore, hyphen)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_name):
            raise UserValidationError("Username can only contain letters, numbers, underscores, and hyphens")

    def create_user(
        self,
        user_name: str,
        hashed_password: str,
        **kwargs: Any
    ) -> uuid.UUID:
        """
        Create a new user.

        Args:
            user_name: Unique username
            hashed_password: Hashed password (should be hashed by caller)
            **kwargs: Additional arguments

        Returns:
            User ID as UUID

        Raises:
            UserValidationError: If validation fails
            StorageOperationError: If storage operation fails
        """
        try:
            # Validate input
            self._validate_user_creation(user_name, hashed_password)

            # Check if username already exists
            existing_user = self.metadata_store.get_user_by_username(user_name, **kwargs)
            if existing_user:
                raise UserValidationError(f"Username '{user_name}' already exists")

            # Create user metadata
            user_metadata = User(
                user_name=user_name,
                hashed_password=hashed_password,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # Store user metadata
            logger.info(f"Creating user: {user_name}")
            stored_user_id = self.metadata_store.store_user(user_metadata, **kwargs)

            if not stored_user_id:
                raise StorageOperationError("Failed to store user metadata")

            logger.info(f"Successfully created user: {user_name} (user_id: {str(stored_user_id)})")
            return stored_user_id

        except UserValidationError:
            raise
        except Exception as e:
            error_msg = f"User creation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StorageOperationError(error_msg)

    def get_user(
        self,
        user_id: uuid.UUID,
        **kwargs: Any
    ) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID as UUID
            **kwargs: Additional arguments

        Returns:
            User metadata or None if not found
        """
        try:
            return self.metadata_store.get_user(user_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None

    def get_user_by_username(
        self,
        username: str,
        **kwargs: Any
    ) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username
            **kwargs: Additional arguments

        Returns:
            User metadata or None if not found
        """
        try:
            return self.metadata_store.get_user_by_username(username, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None

    def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[User]:
        """
        List all users with pagination.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            **kwargs: Additional arguments

        Returns:
            List of user metadata
        """
        try:
            return self.metadata_store.list_users(limit=limit, offset=offset, **kwargs)
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []

    def update_user(
        self,
        user_id: uuid.UUID,
        updates: dict,
        **kwargs: Any
    ) -> bool:
        """
        Update user metadata.

        Args:
            user_id: User ID as UUID
            updates: Dictionary of fields to update
            **kwargs: Additional arguments

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # Validate updates
            if 'user_name' in updates:
                # Check if new username already exists
                new_username = updates['user_name']
                existing_user = self.metadata_store.get_user_by_username(new_username, **kwargs)
                if existing_user and existing_user.id != user_id:
                    raise UserValidationError(f"Username '{new_username}' already exists")

            # Add updated_at timestamp
            updates['updated_at'] = datetime.now()

            logger.info(f"Updating user {user_id}: {list(updates.keys())}")
            success = self.metadata_store.update_user(user_id, updates, **kwargs)

            if success:
                logger.info(f"Successfully updated user {user_id}")
            else:
                logger.warning(f"Failed to update user {user_id}")

            return success

        except UserValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return False

    def delete_user(
        self,
        user_id: uuid.UUID,
        **kwargs: Any
    ) -> bool:
        """
        Delete user.

        Note: This will cascade delete all associated chat sessions and messages
        due to foreign key constraints.

        Args:
            user_id: User ID as UUID
            **kwargs: Additional arguments

        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            logger.info(f"Deleting user {user_id}")
            success = self.metadata_store.delete_user(user_id, **kwargs)

            if success:
                logger.info(f"Successfully deleted user {user_id}")
            else:
                logger.warning(f"Failed to delete user {user_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False

