from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ChunksStatus(Enum):
    """Status of chunks processing"""
    CHUNKING = "chunking"       # Chunking in progress
    CHUNKED = "chunked"         # Successfully chunked
    INDEXED = "indexed"         # Chunks indexed in vector store
    FAILED = "failed"           # Chunking failed
    ARCHIVED = "archived"       # Content archived


@dataclass
class ChunksMetadata:
    """
    Metadata for chunked content tracking in RAG pipeline.

    This model tracks the chunking process applied to parsed content,
    including chunking strategy and the resulting chunks storage information.
    """

    # Primary identifiers
    chunks_id: str                      # Unique ID for this chunks collection
    source_parsed_content_id: str       # Links to ParsedContentMetadata

    # Storage information
    blob_key: str                       # Storage key for chunks JSON file
    chunks_count: int                   # Number of chunks created
    content_size: int                   # Total size of chunks JSON file in bytes
    checksum: str                       # SHA-256 of chunks JSON file

    # Chunking configuration
    chunking_strategy: str              # Strategy used (e.g., "fixed_1000", "semantic_0.8")
    chunking_version: Optional[str]        # Version of chunking algorithm

    # Processing status
    status: ChunksStatus

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Content metadata
    content_type: str = "application/json"  # Always JSON for chunks file

    # Optional processing metadata
    processing_time_ms: Optional[int] = None    # Time taken to chunk
    chunking_config: Optional[str] = None       # Additional JSON config if needed