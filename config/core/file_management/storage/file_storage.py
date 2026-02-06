"""Configuration for FileStorage (Core Layer)"""

from framework.config import AbstractConfig
from core.file_management.storage.file import FileStorage
from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from typing import Literal


class FileStorageConfig(AbstractConfig):
    """Configuration for FileStorage - directly coordinates blob storage with metadata database"""
    type: Literal["file_storage"] = "file_storage"

    # Direct sub-configurations (no intermediate file_store layer)
    file_db_config: LocalDBConfig  # Blob storage configuration (LocalDB or MinIODB)
    relational_db_config: PostgreSQLConfig  # Metadata database configuration

    def build(self) -> FileStorage:
        return FileStorage(self)