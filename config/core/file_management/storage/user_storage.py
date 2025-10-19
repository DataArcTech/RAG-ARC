"""Configuration for UserStorage (Core Layer)"""

from framework.config import AbstractConfig
from core.user_management.user import UserStorage
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from typing import Literal


class UserStorageConfig(AbstractConfig):
    """Configuration for UserStorage - manages user metadata in database"""
    type: Literal["user_storage"] = "user_storage"

    # Database configuration
    relational_db_config: PostgreSQLConfig  # Metadata database configuration

    def build(self) -> UserStorage:
        return UserStorage(self)

