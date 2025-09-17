from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class ParsedContentStatus(Enum):
    """Status of parsed content"""
    UPLOADING = "uploading"
    PARSING = "parsing"
    PARSED = "parsed"
    INDEXED = "indexed"
    FAILED = "failed"
    OUTDATED = "outdated"


@dataclass
class ParsedContentMetadata:
    """Metadata for parsed content (markdown, text, etc.)"""
    parsed_content_id: str
    source_asset_id: str
    blob_key: str
    content_size: int
    checksum: str
    parser_type: str
    parser_version: Optional[str]
    status: ParsedContentStatus
    created_at: datetime
    updated_at: datetime
    content_type: str = "text/markdown"
    parsing_config: Optional[str] = None
    page_count: Optional[int] = None
    language: Optional[str] = None