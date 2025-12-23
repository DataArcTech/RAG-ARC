"""Configuration for Local file storage"""

import os
from framework.config import AbstractConfig
from encapsulation.database.file_db.local import LocalDB
from typing import Literal
from pydantic import Field


class LocalDBConfig(AbstractConfig):
    """Configuration for Local file storage - stores files on local filesystem"""
    # Discriminator for config type identification
    type: Literal["local_blob_store"] = "local_blob_store"

    # Local storage configuration
    base_path: str = Field(
        default_factory=lambda: os.getenv("LOCAL_FILE_STORAGE_PATH")
        or os.getenv("LOCAL_BLOB_STORE_BASE_PATH", "./data/files")
    )
    cleanup_empty_dirs: bool = False  # Whether to remove empty directories on cleanup

    def build(self) -> LocalDB:
        return LocalDB(self)
