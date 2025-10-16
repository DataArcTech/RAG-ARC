"""
Database models for RAG-ARC system.

This module contains all SQLAlchemy ORM models for file metadata, parsed content metadata,
chunks metadata, user management, and chat session management.
All models share a common Base for proper table creation and relationships.
"""

import uuid
import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy import String, DateTime, BigInteger, Integer, Enum as SQLEnum, ForeignKey


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models"""
    pass


# ==================== USER MANAGEMENT ====================

class User(Base):
    """
    User model for account management and authentication.
    Supports JWT-based authentication without cookie table.
    """
    __tablename__ = 'user'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # Relationships
    files: Mapped[List["FileMetadata"]] = relationship(back_populates="owner")
    chat_sessions: Mapped[List["ChatSession"]] = relationship(back_populates="user")


# ==================== CHAT SESSION MANAGEMENT ====================

class ChatSession(Base):
    """
    Chat session model for managing user conversations.

    Supports session recovery and conversation history management.
    """
    __tablename__ = 'chat_session'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="chat_sessions")
    messages: Mapped[List["ChatMessage"]] = relationship(back_populates="session")


class ChatMessage(Base):
    """
    Chat message model for storing conversation messages.

    Uses JSON field to store structured message content (role, content, metadata, etc.).
    """
    __tablename__ = 'chat_message'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_session.id"), nullable=False, index=True
    )

    # JSON content structure: {"role": "user/assistant/system", "content": "...", "metadata": {...}}
    content: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )

    # Relationship
    session: Mapped["ChatSession"] = relationship(back_populates="messages")


# ==================== FILE METADATA ====================

class FileStatus(enum.Enum):
    """File processing status"""
    STORED = "STORED"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    DELETED = "DELETED"


@dataclass
class FileMetadata(Base):
    """
    File metadata model for tracking file assets.
    Stores file metadata with references to object storage.
    """
    __tablename__ = 'file_metadata'

    # Primary identifier
    file_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # User info
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)

    # Storage information
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # Processing status
    status: Mapped[FileStatus] = mapped_column(SQLEnum(FileStatus), nullable=False)

    # File properties
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Relationships
    parsed_contents: Mapped[List["ParsedContentMetadata"]] = relationship(
        back_populates="source_file"
    )
    owner: Mapped["User"] = relationship(back_populates="files")


# ==================== PARSED CONTENT METADATA ====================

class ParsedContentStatus(enum.Enum):
    """Status of parsed content"""
    STORED = "STORED"
    CHUNKED = "CHUNKED"
    FAILED = "FAILED"


@dataclass
class ParsedContentMetadata(Base):
    """
    Parsed content metadata model for storing parsed file content metadata.
    Tracks parsed content (markdown, text, etc.) in the RAG pipeline.
    """
    __tablename__ = 'parsed_content_metadata'

    # Primary identifiers
    parsed_content_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_file_id: Mapped[str] = mapped_column(String(255), ForeignKey("file_metadata.file_id"), nullable=False)

    # Storage information
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)

    # Parser information
    parser_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Processing status
    status: Mapped[ParsedContentStatus] = mapped_column(SQLEnum(ParsedContentStatus), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Content metadata
    content_type: Mapped[str] = mapped_column(String(100), default="text/markdown", nullable=False)

    # Relationships
    source_file: Mapped["FileMetadata"] = relationship(back_populates="parsed_contents")
    chunks: Mapped[List["ChunkMetadata"]] = relationship(
        back_populates="source_parsed_content"
    )


# ==================== CHUNK METADATA ====================

class ChunkIndexStatus(enum.Enum):
    """Status of individual chunk indexing"""
    STORED = "STORED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


@dataclass
class ChunkMetadata(Base):
    """
    Chunk metadata model for individual chunks created from parsed content.
    Tracks chunk storage and indexing status in the vector store.
    """
    __tablename__ = 'chunk_metadata'

    # Primary identifier
    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Foreign key to source parsed content
    source_parsed_content_id: Mapped[str] = mapped_column(String(255), ForeignKey("parsed_content_metadata.parsed_content_id"), nullable=False)

    # Owner information (for user isolation in retrieval)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True)

    # File storage reference
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)

    # Chunking configuration
    chunker_type: Mapped[str] = mapped_column(String(100), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Index of chunk within the parsed content

    # Indexing status
    index_status: Mapped[ChunkIndexStatus] = mapped_column(SQLEnum(ChunkIndexStatus), default=ChunkIndexStatus.STORED, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationship
    source_parsed_content: Mapped["ParsedContentMetadata"] = relationship(back_populates="chunks")
    owner: Mapped["User"] = relationship()