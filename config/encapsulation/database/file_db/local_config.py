"""Configuration for Local file storage"""

from framework.config import AbstractConfig
from encapsulation.database.file_db.local import LocalDB
from typing import Literal


class LocalDBConfig(AbstractConfig):
    """Configuration for Local file storage - stores files on local filesystem"""
    # Discriminator for config type identification
    type: Literal["local_blob_store"] = "local_blob_store"

    # Local storage configuration
    base_path: str = "./test_output"  # Base directory for file storage
    cleanup_empty_dirs: bool = False  # Whether to remove empty directories on cleanup

    def build(self) -> LocalDB:
        return LocalDB(self)