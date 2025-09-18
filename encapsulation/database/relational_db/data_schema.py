"""
Database models for RAG-ARC system.

This module contains all SQLAlchemy ORM models for file metadata, parsed content metadata,
and chunks metadata, sharing a common Base for proper table creation and relationships.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, DateTime, BigInteger, Text, Enum as SQLEnum

Base = declarative_base()


# ==================== FILE METADATA ====================

class FileStatus(Enum):
    """File processing status"""
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    PARSING = "PARSING"
    PARSED = "PARSED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


@dataclass
class FileMetadata(Base):
    """
    File metadata model for tracking file assets.

    Combines SQLAlchemy ORM capabilities for file storage and metadata
    management in the RAG pipeline.
    """

    __tablename__ = 'file_metadata'

    # Primary identifier
    asset_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Storage information
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # Processing status (automatically handles enum conversion)
    status: Mapped[FileStatus] = mapped_column(SQLEnum(FileStatus), nullable=False)

    # File properties
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100))
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Optional metadata
    original_path: Mapped[Optional[str]] = mapped_column(Text)


# ==================== PARSED CONTENT METADATA ====================

class ParsedContentStatus(Enum):
    """Status of parsed content"""
    UPLOADING = "UPLOADING"
    PARSING = "PARSING"
    PARSED = "PARSED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    OUTDATED = "OUTDATED"


@dataclass
class ParsedContentMetadata(Base):
    """
    Metadata for parsed content (markdown, text, etc.).

    Combines SQLAlchemy ORM capabilities for parsed content management
    in the RAG pipeline.
    """

    __tablename__ = 'parsed_content_metadata'

    # Primary identifiers
    parsed_content_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_asset_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Storage information
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)
    content_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    # Parser information
    parser_type: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Processing status (automatically handles enum conversion)
    status: Mapped[ParsedContentStatus] = mapped_column(SQLEnum(ParsedContentStatus), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Content metadata
    content_type: Mapped[str] = mapped_column(String(100), default="text/markdown", nullable=False)

    # Optional parsing metadata
    parsing_config: Mapped[Optional[str]] = mapped_column(Text)
    page_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    language: Mapped[Optional[str]] = mapped_column(String(10))


# ==================== CHUNKS METADATA ====================

class ChunksStatus(Enum):
    """Status of chunks processing"""
    CHUNKING = "CHUNKING"       # Chunking in progress
    CHUNKED = "CHUNKED"         # Successfully chunked
    INDEXED = "INDEXED"         # Chunks indexed in vector store
    FAILED = "FAILED"           # Chunking failed
    ARCHIVED = "ARCHIVED"       # Content archived


@dataclass
class ChunksMetadata(Base):
    """
    Metadata for chunked content tracking in RAG pipeline.

    This model tracks the chunking process applied to parsed content,
    including chunking strategy and the resulting chunks storage information.
    Combines SQLAlchemy ORM capabilities for chunks management.
    """

    __tablename__ = 'chunks_metadata'

    # Primary identifiers
    chunks_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_parsed_content_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Storage information
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)
    chunks_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    # Chunking configuration
    chunking_strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    chunking_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Processing status (automatically handles enum conversion)
    status: Mapped[ChunksStatus] = mapped_column(SQLEnum(ChunksStatus), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Content metadata
    content_type: Mapped[str] = mapped_column(String(100), default="application/json", nullable=False)

    # Optional processing metadata
    processing_time_ms: Mapped[Optional[int]] = mapped_column(BigInteger)
    chunking_config: Mapped[Optional[str]] = mapped_column(Text)

    # Indexing metadata (only populated when status = INDEXED)
    index_type: Mapped[Optional[str]] = mapped_column(String(50))