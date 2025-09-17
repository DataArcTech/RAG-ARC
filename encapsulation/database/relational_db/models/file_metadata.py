from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class FileStatus(Enum):
    """File processing status"""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class FileMetadata:
    """File metadata model for tracking file assets"""
    asset_id: str
    blob_key: str
    filename: str
    status: FileStatus
    file_size: int
    content_type: str
    checksum: str
    created_at: datetime
    updated_at: datetime
    original_path: Optional[str] = None