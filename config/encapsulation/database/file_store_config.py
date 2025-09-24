"""Configuration for coordinated FileStore"""

from framework.config import AbstractConfig
from encapsulation.database.file_store import FileStore
from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from typing import Literal


class FileStoreConfig(AbstractConfig):
    """Configuration for coordinated FileStore - combines blob storage with metadata database"""
    # Discriminator for config type identification
    type: Literal["file_store"] = "file_store"

    # Required sub-configurations
    file_db_config: LocalDBConfig  # Blob storage configuration (LocalDB or MinIODB)
    relational_db_config: PostgreSQLConfig  # Metadata database configuration

    def build(self) -> FileStore:
        return FileStore(self)