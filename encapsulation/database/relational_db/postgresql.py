from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    List,
    Dict,
)
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, Engine, text, inspect as sqlalchemy_inspect
from sqlalchemy.orm import sessionmaker, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.dialects.postgresql import JSON

from .base import RelationalDB
from .schema_patches import apply_postgres_schema_patches
from ...data_model.orm_models import (
    Base,
    User, UserStatus,
    Department,
    Role, RoleType,
    ChatSession,
    ChatMessage,
    FileMetadata, FileStatus,
    FilePermission, PermissionReceiverType, PermissionType,
    AuditLog, AuditAction,
    ParsedContentMetadata, ParsedContentStatus,
    ChunkMetadata, ChunkIndexStatus,
    TaskRun, TaskRunState,
    TaskProgressEvent,
    TaskSyncOffset,
)
from framework.singleton_decorator import singleton

from .postgresql_chat import _PostgreSQLChatMixin
from .postgresql_chunks import _PostgreSQLChunksMixin
from .postgresql_files import _PostgreSQLFilesMixin
from .postgresql_parsed_content import _PostgreSQLParsedContentMixin
from .postgresql_permissions import _PostgreSQLPermissionsMixin
from .postgresql_users import _PostgreSQLUsersMixin

logger = logging.getLogger(__name__)


@singleton
class PostgreSQLDB(
    _PostgreSQLFilesMixin,
    _PostgreSQLParsedContentMixin,
    _PostgreSQLChunksMixin,
    _PostgreSQLUsersMixin,
    _PostgreSQLChatMixin,
    _PostgreSQLPermissionsMixin,
    RelationalDB,
):
    """
    PostgreSQL implementation for relational database operations with hybrid SQLAlchemy/raw SQL approach.
    
    This class provides a complete relational database solution using PostgreSQL, combining
    SQLAlchemy for schema management and connection handling with raw SQL for high-performance
    data operations. It supports file metadata, parsed content metadata, and extensible
    schema for future user management and application data.
    
    Key features:
    - Hybrid approach: SQLAlchemy for schema, raw SQL for operations
    - Automatic database and table creation
    - Connection pooling with configurable pool sizes
    - Comprehensive error handling with detailed logging
    - Support for file and parsed content metadata
    - Transactional operations with proper rollback
    - Extensible schema design for future requirements
    
    Storage architecture:
    - file_metadata table: Core file information and status tracking
    - parsed_content_metadata table: Parsed content relationships and metadata
    - UUID-based primary keys for global uniqueness
    - Timestamp tracking for created_at/updated_at
    - Status enums for workflow tracking
    
    Performance considerations:
    - Connection pooling reduces connection overhead (default: 10 connections)
    - Raw SQL operations minimize ORM overhead for data manipulation
    - Indexes on primary keys and foreign keys for fast lookups
    - Pre-ping enabled to handle stale connections
    - Optional SQL query logging for debugging
    
    Configuration parameters:
        host (str): PostgreSQL server hostname
        port (int): PostgreSQL server port (default: 5432)
        database (str): Target database name
        user (str): Database username
        password (str): Database password
        pool_size (int): Connection pool size (default: 10)
        max_overflow (int): Maximum overflow connections (default: 20)
        echo_sql (bool): Enable SQL query logging (default: False)
        
    Schema management:
    - Automatic database creation if not exists
    - SQLAlchemy models define table structure
    - Automatic table creation via Base.metadata.create_all()
    - Future extensibility for user tables, permissions, etc.
    
    Transaction handling:
    - Each operation uses a connection context manager
    - Automatic commit on success, rollback on exception
    - Integrity errors are caught and converted to ValueError
    - Connection cleanup handled automatically
    
    Typical usage:
        >>> config = PostgreSQLConfig(
        ...     host="localhost",
        ...     database="myapp",
        ...     user="postgres", 
        ...     password="password"
        ... )
        >>> db = PostgreSQLDB(config)
        >>> metadata_id = db.store_file_metadata(file_metadata)
        >>> metadata = db.get_file_metadata(metadata_id)
        
    Error handling:
    - IntegrityError mapped to ValueError for duplicate keys
    - SQLAlchemyError logged with full context
    - Connection errors handled with appropriate retries
    - Database creation errors logged and re-raised
        
    Attributes:
        config: Configuration object with PostgreSQL connection parameters
        _engine: Cached SQLAlchemy engine instance (lazy-initialized)
    """
    
    def __init__(self, config):
        """Initialize PostgreSQL with eager engine and session creation"""
        super().__init__(config)
        # Build engine and session maker immediately since we always need them
        self.engine = self._create_engine()
        self.SessionMaker = sessionmaker(bind=self.engine)

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine"""
        # First, try to create database if it doesn't exist
        self._ensure_database_exists()

        # Build connection string for psycopg3
        connection_string = (
            f"postgresql+psycopg://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{int(getattr(self.config, 'port', '5432'))}/{self.config.database}"
        )

        engine = create_engine(
            connection_string,
            pool_size=getattr(self.config, 'pool_size', 10),
            max_overflow=getattr(self.config, 'max_overflow', 20),
            pool_pre_ping=True,
            echo=getattr(self.config, 'echo_sql', False)
        )

        # Create tables using SQLAlchemy
        Base.metadata.create_all(engine)
        apply_postgres_schema_patches(engine)
        self._auto_migrate_missing_columns(engine)
        logger.info("PostgreSQL engine initialized and tables created")

        return engine

    def _auto_migrate_missing_columns(self, engine: Engine) -> None:
        """Best-effort additive schema migration for missing nullable columns.

        Notes:
        - `Base.metadata.create_all()` does not add new columns to existing tables.
        - This helper is intentionally conservative:
          - Only adds columns that are nullable in ORM (to avoid breaking existing rows).
          - Never drops columns or changes types.
        """

        try:
            inspector = sqlalchemy_inspect(engine)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping auto-migration: unable to create inspector (%s)", exc)
            return

        migrated_count = 0
        for table_name, table in Base.metadata.tables.items():
            try:
                if not inspector.has_table(table_name):
                    continue
                existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping auto-migration for table %s (%s)", table_name, exc)
                continue

            for column in table.columns:
                column_name = column.name
                if column_name in existing_columns:
                    continue

                if not column.nullable:
                    logger.info(
                        "Skipping non-nullable missing column %s.%s; use an explicit migration.",
                        table_name,
                        column_name,
                    )
                    continue

                try:
                    column_type = column.type
                    if isinstance(column_type, JSON):
                        sql_type = "JSON"
                    else:
                        sql_type = column_type.compile(dialect=engine.dialect)

                    alter_sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {sql_type}'
                    with engine.begin() as conn:
                        conn.execute(text(alter_sql))
                    migrated_count += 1
                    logger.info("Added missing column: %s.%s (%s)", table_name, column_name, sql_type)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to add column %s.%s (%s)", table_name, column_name, exc)

        if migrated_count:
            logger.info("Auto-migration completed; added %d columns.", migrated_count)
    
    def _ensure_database_exists(self) -> None:
        """Create database if it doesn't exist"""
        try:
            # Connect to postgres database to create our target database
            admin_connection_string = (
                f"postgresql+psycopg://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{int(getattr(self.config, 'port', '5432'))}/postgres"
            )
            
            admin_engine = create_engine(admin_connection_string, isolation_level="AUTOCOMMIT")
            
            with admin_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(text(
                    "SELECT 1 FROM pg_database WHERE datname = :database"
                ), {"database": self.config.database})
                
                if not result.fetchone():
                    # Database doesn't exist, create it
                    conn.execute(text(f'CREATE DATABASE "{self.config.database}"'))
                    logger.info(f"Created database: {self.config.database}")
                else:
                    logger.debug(f"Database {self.config.database} already exists")
                    
            admin_engine.dispose()
            
        except SQLAlchemyError as e:
            logger.error(f"Error ensuring database exists: {e}")
            raise
    
