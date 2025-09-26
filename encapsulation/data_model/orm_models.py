"""
Database models for RAG-ARC system.

This module contains all SQLAlchemy ORM models for file metadata, parsed content metadata,
and chunks metadata, sharing a common Base for proper table creation and relationships.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import String, DateTime, BigInteger, Text, Enum as SQLEnum, ForeignKey

Base = declarative_base()


# ==================== FILE METADATA ====================

class FileStatus(Enum):
    """File processing status"""
    STORED = "STORED"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"    # Mark as FAILED when error occurs and wait for validation


@dataclass
class FileMetadata(Base):
    """
    File metadata model for tracking file assets.

    Combines SQLAlchemy ORM capabilities for file storage and metadata
    management in the RAG pipeline.
    """

    __tablename__ = 'file_metadata'

    # Primary identifier
    file_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # User info
    # owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)

    # Storage information
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # Processing status (automatically handles enum conversion)
    status: Mapped[FileStatus] = mapped_column(SQLEnum(FileStatus), nullable=False)

    # File properties
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Relationships
    parsed_contents: Mapped[List["ParsedContentMetadata"]] = relationship(back_populates="source_file")

    # User who owns this file 
    # owner: Mapped["User"] = relationship(back_populates="file_metadata")


# ==================== PARSED CONTENT METADATA ====================

class ParsedContentStatus(Enum):
    """Status of parsed content"""
    STORED = "STORED"    
    CHUNKED = "CHUNKED"
    FAILED = "FAILED"    # Mark as FAILED when error occurs and wait for validation


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
    source_file_id: Mapped[str] = mapped_column(String(255), ForeignKey("file_metadata.file_id"), nullable=False)

    # Storage information
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)

    # Parser information
    parser_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Processing status (automatically handles enum conversion)
    status: Mapped[ParsedContentStatus] = mapped_column(SQLEnum(ParsedContentStatus), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Content metadata
    content_type: Mapped[str] = mapped_column(String(100), default="text/markdown", nullable=False)

    # Relationships
    source_file: Mapped["FileMetadata"] = relationship(back_populates="parsed_contents")
    chunks: Mapped[List["ChunkMetadata"]] = relationship(back_populates="source_parsed_content")


# ==================== INDIVIDUAL CHUNK METADATA ====================

class ChunkIndexStatus(Enum):
    """Status of individual chunk indexing"""
    STORED = "STORED"           # Stored in file_db
    INDEXED = "INDEXED"         # Successfully indexed in vector store
    FAILED = "FAILED"           # Failed to index


@dataclass
class ChunkMetadata(Base):
    """
    Metadata for individual chunks created from parsed content.

    This model tracks each individual chunk with its own JSON file storage
    in file_db and indexing status in the vector store for precise tracking
    of the indexing pipeline progress.
    """

    __tablename__ = 'chunk_metadata'

    # Primary identifier
    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Foreign key to source parsed content
    source_parsed_content_id: Mapped[str] = mapped_column(String(255), ForeignKey("parsed_content_metadata.parsed_content_id"), nullable=False)

    # File storage reference
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)  # Reference to JSON file in file_db

    # Chunking configuration used for this chunk
    chunker_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Indexing status
    index_status: Mapped[ChunkIndexStatus] = mapped_column(SQLEnum(ChunkIndexStatus), default=ChunkIndexStatus.STORED, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)  # When successfully indexed

    # Relationship back to source parsed content
    source_parsed_content: Mapped["ParsedContentMetadata"] = relationship(back_populates="chunks")