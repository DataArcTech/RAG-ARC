"""Configuration for coordinated FileStore"""

from framework.config import AbstractConfig
from core.file_management.file_storage import FileStorage
from config.encapsulation.database.file_store_config import FileStoreConfig
from typing import Literal


class FileStorageConfig(AbstractConfig):
    """Configuration for FileStorage"""
    type: Literal["file_storage"] = "file_storage"
    data_store_config: FileStoreConfig

    def build(self) -> FileStorage:
        return FileStorage(self)