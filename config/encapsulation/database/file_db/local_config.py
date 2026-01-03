"""Configuration for Local file storage"""

import os
from pathlib import Path
from framework.config import AbstractConfig
from encapsulation.database.file_db.local import LocalDB
from typing import Literal
from pydantic import Field

from core.utils.filename_guard import project_root_dir


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
        base = Path(str(self.base_path or "")).expanduser()
        if not base.is_absolute():
            base = (project_root_dir() / base).resolve()
        normalized = self.model_copy(update={"base_path": str(base)})
        return LocalDB(normalized)
