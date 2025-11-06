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
from sqlalchemy import String, DateTime, BigInteger, Integer, Boolean, Text, Enum as SQLEnum, ForeignKey


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models"""
    pass


# ==================== ENUMS ====================

# Enum to indicate permission target type (user, department, or all)
class PermissionReceiverType(enum.Enum):
    USER = "user"
    DEPARTMENT = "department"
    ALL = "all"

class PermissionType(enum.Enum):
    VIEW = "view"
    EDIT = "edit"

class UserStatus(enum.Enum):
    """User account status"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"


class RoleType(enum.Enum):
    """System-wide role types"""
    SUPER_ADMIN = "SUPER_ADMIN"  # Full system access
    DEPT_ADMIN = "DEPT_ADMIN"    # Department administrator
    USER = "USER"                 # Regular user
    GUEST = "GUEST"              # Read-only guest


class PermissionAction(enum.Enum):
    """Available permission actions"""
    VIEW = "VIEW"
    EDIT = "EDIT"
    MANAGE = "MANAGE"  # Includes permission management

class AuditAction(enum.Enum):
    """Audit log action types"""
    FILE_UPLOAD = "FILE_UPLOAD"
    FILE_VIEW = "FILE_VIEW"
    FILE_EDIT = "FILE_EDIT"
    FILE_DELETE = "FILE_DELETE"
    FILE_SHARE = "FILE_SHARE"
    FILE_UNSHARE = "FILE_UNSHARE"
    PERMISSION_GRANT = "PERMISSION_GRANT"
    PERMISSION_REVOKE = "PERMISSION_REVOKE"
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"


# ==================== DEPARTMENT ====================

class Department(Base):
    """
    Department model for organizational hierarchy.
    Supports tree structure with parent-child relationships.
    """
    __tablename__ = 'department'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    parent_department_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("department.id"), index=True
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Tree path for efficient querying (e.g., "dept1/dept2")
    path: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationships
    parent_department: Mapped[Optional["Department"]] = relationship(
        remote_side=[id], back_populates="child_departments"
    )
    child_departments: Mapped[List["Department"]] = relationship(
        back_populates="parent_department", cascade="all, delete-orphan"
    )
    members: Mapped[List["User"]] = relationship(back_populates="department")
    # Files shared with this department, could be either view or edit permission
    file_permissions: Mapped[List["FilePermission"]] = relationship(
        back_populates="department"
    )


# ==================== USER MANAGEMENT ====================

class User(Base):
    """
    Enhanced User model with department and role support.
    Each user belongs to exactly one department.
    """
    __tablename__ = 'user'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    department_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("department.id"), index=True
    )
    role_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("role.id"), index=True
    )

    user_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Status
    status: Mapped[UserStatus] = mapped_column(
        SQLEnum(UserStatus), default=UserStatus.ACTIVE, nullable=False
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    department: Mapped[Optional["Department"]] = relationship(back_populates="members")
    role: Mapped[Optional["Role"]] = relationship(back_populates="users")
    files: Mapped[List["FileMetadata"]] = relationship(back_populates="owner")
    chat_sessions: Mapped[List["ChatSession"]] = relationship(back_populates="user")
    # Files shared with this user, could be either view or edit permission
    file_permissions: Mapped[List["FilePermission"]] = relationship(
        "FilePermission",
        foreign_keys="[FilePermission.user_id]",
        back_populates="user",
        primaryjoin="User.id == FilePermission.user_id"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(back_populates="user")


# ==================== ROLE & PERMISSION ====================

class Role(Base):
    """
    Role model for RBAC implementation.
    System-wide roles for permission management.
    """
    __tablename__ = 'role'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    role_type: Mapped[RoleType] = mapped_column(SQLEnum(RoleType), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Permissions as JSON (flexible for future expansion)
    # Format: {"files": ["view", "edit"], "users": ["view"], ...}
    permissions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationships
    users: Mapped[List["User"]] = relationship(back_populates="role")


# ==================== CHAT SESSION MANAGEMENT ====================

class ChatSession(Base):
    """
    Enhanced chat session with user context.
    """
    __tablename__ = 'chat_session'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Session metadata
    is_shared: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    session_metadata: Mapped[Optional[dict]] = mapped_column("session_metadata", JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="chat_sessions")
    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """
    Chat message model with enhanced tracking.
    """
    __tablename__ = 'chat_message'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_session.id"), nullable=False, index=True
    )

    # JSON content structure
    content: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Source tracking for RAG
    source_file_ids: Mapped[Optional[list]] = mapped_column(JSON)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    # Relationship
    session: Mapped["ChatSession"] = relationship(back_populates="messages")


# ==================== FILE METADATA WITH PERMISSIONS ====================

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
    Enhanced File metadata with visibility and permission support.
    """
    __tablename__ = 'file_metadata'

    # Primary identifier
    file_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Owner
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True
    )

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
    owner: Mapped["User"] = relationship(back_populates="files")
    parsed_contents: Mapped[List["ParsedContentMetadata"]] = relationship(
        back_populates="source_file", cascade="all, delete-orphan"
    )
    permissions: Mapped[List["FilePermission"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(back_populates="file")


class FilePermission(Base):
    """
        Explicit file permissions for users/departments/all.
        Used when files are shared with specific users/departments/all.
    """
    __tablename__ = 'file_permission'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("file_metadata.file_id"), nullable=False, index=True
    )

    permission_receiver_type: Mapped["PermissionReceiverType"] = mapped_column(
        SQLEnum(PermissionReceiverType), nullable=False, default=PermissionReceiverType.USER
    )

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=True, index=True
    )
    department_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("department.id"), nullable=True, index=True
    )

    # Permission type (view, edit)
    permission_type: Mapped["PermissionType"] = mapped_column(
        SQLEnum(PermissionType), nullable=False, default=PermissionType.VIEW
    )

    # Grant information
    granted_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=False
    )
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    # Relationships
    file: Mapped["FileMetadata"] = relationship(back_populates="permissions")
    user: Mapped["User"] = relationship(
        foreign_keys=[user_id], back_populates="file_permissions"
    )
    department: Mapped["Department"] = relationship(back_populates="file_permissions")

# ==================== AUDIT LOG ====================

class AuditLog(Base):
    """
    Audit log for tracking all user actions and file operations.
    Essential for compliance and security.
    """
    __tablename__ = 'audit_log'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True
    )

    # Action details
    action: Mapped[AuditAction] = mapped_column(SQLEnum(AuditAction), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'file', 'user', 'permission', etc.
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)

    # Optional file reference
    file_id: Mapped[Optional[str]] = mapped_column(
        String(255), ForeignKey("file_metadata.file_id"), index=True
    )

    # Action metadata (flexible JSON field)
    # Format: {"ip": "192.168.1.1", "user_agent": "...", "details": {...}}
    log_metadata: Mapped[dict] = mapped_column("log_metadata", JSON, default=dict, nullable=False)

    # Status and result
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False, index=True
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="audit_logs")
    file: Mapped[Optional["FileMetadata"]] = relationship(back_populates="audit_logs")


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
        back_populates="source_parsed_content", cascade="all, delete-orphan"
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
    Enhanced chunk metadata with owner context.
    """
    __tablename__ = 'chunk_metadata'

    # Primary identifier
    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Foreign key to source parsed content
    source_parsed_content_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("parsed_content_metadata.parsed_content_id"), nullable=False
    )

    # Owner information (for user isolation in retrieval)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True
    )

    # File storage reference
    blob_key: Mapped[str] = mapped_column(String(500), nullable=False)

    # Chunking configuration
    chunker_type: Mapped[str] = mapped_column(String(100), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Indexing status
    index_status: Mapped[ChunkIndexStatus] = mapped_column(
        SQLEnum(ChunkIndexStatus), default=ChunkIndexStatus.STORED, nullable=False
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    source_parsed_content: Mapped["ParsedContentMetadata"] = relationship(back_populates="chunks")
    owner: Mapped["User"] = relationship()