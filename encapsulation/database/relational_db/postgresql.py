from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    List,
    Dict,
)
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .base import RelationalDB
from ...data_model.orm_models import (
    Base,
    FileMetadata, FileStatus,
    ParsedContentMetadata, ParsedContentStatus,
    ChunkMetadata, ChunkIndexStatus
)
from framework.singleton_decorator import singleton

logger = logging.getLogger(__name__)


@singleton
class PostgreSQLDB(RelationalDB):
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
            f"@{self.config.host}:{getattr(self.config, 'port', 5432)}/{self.config.database}"
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
        logger.info("PostgreSQL engine initialized and tables created")

        return engine
    
    def _ensure_database_exists(self) -> None:
        """Create database if it doesn't exist"""
        try:
            # Connect to postgres database to create our target database
            admin_connection_string = (
                f"postgresql+psycopg://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{getattr(self.config, 'port', 5432)}/postgres"
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
    
    def store_file_metadata(
        self,
        file_metadata: FileMetadata,
        **kwargs: Any,
    ) -> str:
        """Store file metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                session.add(file_metadata)
                session.commit()
                logger.debug(f"Stored file metadata for asset: {file_metadata.file_id}")
                return file_metadata.file_id

        except IntegrityError:
            logger.error(f"File metadata with file_id '{file_metadata.file_id}' already exists")
            raise ValueError(f"File metadata with file_id '{file_metadata.file_id}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing file metadata: {e}")
            raise
    
    def get_file_metadata(self, file_id: str, **kwargs: Any) -> Optional[FileMetadata]:
        """Retrieve file metadata by file ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                file_metadata = session.query(FileMetadata).filter_by(file_id=file_id).first()

                if file_metadata:
                    logger.debug(f"Retrieved file metadata for file: {file_id}")
                    return file_metadata

                return None

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving file metadata: {e}")
            raise
    
    def update_file_metadata(
        self,
        file_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update file metadata using SQLAlchemy ORM"""
        if not updates:
            return True

        try:
            with self.SessionMaker() as session:
                # Add updated_at timestamp
                updates['updated_at'] = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

                # Update the record (SQLAlchemy handles enum conversion automatically)
                rows_updated = session.query(FileMetadata).filter_by(file_id=file_id).update(updates)
                session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated file metadata for file: {file_id}")
                    return True

                logger.warning(f"No file metadata found to update for file: {file_id}")
                return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating file metadata: {e}")
            raise
    
    def delete_file_metadata(self, file_id: str, **kwargs: Any) -> bool:
        """Delete file metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                rows_deleted = session.query(FileMetadata).filter_by(file_id=file_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.debug(f"Deleted file metadata for file: {file_id}")
                    return True

                logger.warning(f"No file metadata found to delete for file: {file_id}")
                return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting file metadata: {e}")
            raise
    
    def update_file_status(
        self,
        file_id: str,
        new_status: FileStatus,
        **kwargs: Any,
    ) -> bool:
        """Update file processing status"""
        return self.update_file_metadata(
            file_id,
            {'status': new_status},
            **kwargs
        )
    
    def list_file_metadata(
        self,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[FileMetadata]:
        """List file metadata with optional filtering using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                query = session.query(FileMetadata)

                # Add status filter
                if status:
                    query = query.filter(FileMetadata.status == status.value)

                # Add ordering
                query = query.order_by(FileMetadata.created_at.desc())

                # Add pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                file_metadata_list = query.all()
                logger.debug(f"Retrieved {len(file_metadata_list)} file metadata records")

                return file_metadata_list

        except SQLAlchemyError as e:
            logger.error(f"Database error listing file metadata: {e}")
            raise
    
    def store_parsed_content_metadata(
        self,
        parsed_metadata: ParsedContentMetadata,
        **kwargs: Any,
    ) -> str:
        """Store parsed content metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                session.add(parsed_metadata)
                session.commit()
                logger.debug(f"Stored parsed content metadata: {parsed_metadata.parsed_content_id}")
                return parsed_metadata.parsed_content_id

        except IntegrityError:
            logger.error(f"Parsed content metadata with ID '{parsed_metadata.parsed_content_id}' already exists")
            raise ValueError(f"Parsed content metadata with ID '{parsed_metadata.parsed_content_id}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing parsed content metadata: {e}")
            raise
    
    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional[ParsedContentMetadata]:
        """Retrieve parsed content metadata by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                parsed_metadata = session.query(ParsedContentMetadata).filter_by(parsed_content_id=parsed_content_id).first()

                if parsed_metadata:
                    logger.debug(f"Retrieved parsed content metadata: {parsed_content_id}")
                    return parsed_metadata

                return None

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving parsed content metadata: {e}")
            raise
    
    def update_parsed_content_metadata(
        self,
        parsed_content_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update parsed content metadata using SQLAlchemy ORM"""
        if not updates:
            return True

        try:
            with self.SessionMaker() as session:
                # Add updated_at timestamp
                updates['updated_at'] = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

                # Update the record (SQLAlchemy handles enum conversion automatically)
                rows_updated = session.query(ParsedContentMetadata).filter_by(parsed_content_id=parsed_content_id).update(updates)
                session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated parsed content metadata: {parsed_content_id}")
                    return True

                logger.warning(f"No parsed content metadata found to update: {parsed_content_id}")
                return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating parsed content metadata: {e}")
            raise
    
    def delete_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Delete parsed content metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                rows_deleted = session.query(ParsedContentMetadata).filter_by(parsed_content_id=parsed_content_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.debug(f"Deleted parsed content metadata: {parsed_content_id}")
                    return True

                logger.warning(f"No parsed content metadata found to delete: {parsed_content_id}")
                return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting parsed content metadata: {e}")
            raise
    
    def update_parsed_content_status(
        self,
        parsed_content_id: str,
        new_status: ParsedContentStatus,
        **kwargs: Any,
    ) -> bool:
        """Update parsed content processing status"""
        return self.update_parsed_content_metadata(
            parsed_content_id,
            {'status': new_status},
            **kwargs
        )
    
    def list_parsed_content_metadata(
        self,
        source_file_id: Optional[str] = None,
        status: Optional[ParsedContentStatus] = None,
        parser_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ParsedContentMetadata]:
        """List parsed content metadata with optional filtering using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                query = session.query(ParsedContentMetadata)

                # Add filters
                if source_file_id:
                    query = query.filter(ParsedContentMetadata.source_file_id == source_file_id)
                if status:
                    query = query.filter(ParsedContentMetadata.status == status.value)
                if parser_type:
                    query = query.filter(ParsedContentMetadata.parser_type == parser_type)

                # Add ordering
                query = query.order_by(ParsedContentMetadata.created_at.desc())

                # Add pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                parsed_metadata_list = query.all()
                logger.debug(f"Retrieved {len(parsed_metadata_list)} parsed content metadata records")

                return parsed_metadata_list

        except SQLAlchemyError as e:
            logger.error(f"Database error listing parsed content metadata: {e}")
            raise

    # ==================== CHUNK METADATA METHODS ====================

    def store_chunk_metadata(
        self,
        chunk_metadata: ChunkMetadata,
        **kwargs: Any,
    ) -> str:
        """Store chunk metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                session.add(chunk_metadata)
                session.commit()
                logger.info(f"Stored chunk metadata: {chunk_metadata.chunk_id}")
                return chunk_metadata.chunk_id

        except IntegrityError as e:
            if "already exists" in str(e) or "duplicate key" in str(e):
                raise ValueError(f"Chunk with ID {chunk_metadata.chunk_id} already exists")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error storing chunk metadata: {e}")
            raise

    def get_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional[ChunkMetadata]:
        """Get chunk metadata by chunk_id using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                chunk_metadata = session.query(ChunkMetadata).filter_by(chunk_id=chunk_id).first()

                if chunk_metadata:
                    return chunk_metadata
                return None

        except SQLAlchemyError as e:
            logger.error(f"Database error getting chunk metadata for {chunk_id}: {e}")
            raise

    def update_chunk_metadata(
        self,
        chunk_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update chunk metadata fields using SQLAlchemy ORM"""
        if not updates:
            return False

        try:
            with self.SessionMaker() as session:
                # Update the record (SQLAlchemy handles enum conversion automatically)
                rows_updated = session.query(ChunkMetadata).filter_by(chunk_id=chunk_id).update(updates)
                session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated chunk metadata: {chunk_id}")
                    return True
                else:
                    logger.warning(f"No chunk found with ID: {chunk_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating chunk metadata {chunk_id}: {e}")
            raise

    def delete_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> bool:
        """Delete chunk metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                rows_deleted = session.query(ChunkMetadata).filter_by(chunk_id=chunk_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted chunk metadata: {chunk_id}")
                    return True
                else:
                    logger.warning(f"No chunk found with ID: {chunk_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chunk metadata {chunk_id}: {e}")
            raise

    def list_chunk_metadata(
        self,
        source_parsed_content_id: Optional[str] = None,
        index_status: Optional[ChunkIndexStatus] = None,
        chunker_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ChunkMetadata]:
        """List chunk metadata with optional filtering using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                query = session.query(ChunkMetadata)

                # Add filters
                if source_parsed_content_id:
                    query = query.filter(ChunkMetadata.source_parsed_content_id == source_parsed_content_id)
                if index_status:
                    query = query.filter(ChunkMetadata.index_status == index_status.value)
                if chunker_type:
                    query = query.filter(ChunkMetadata.chunker_type == chunker_type)

                # Add ordering
                query = query.order_by(ChunkMetadata.created_at.desc())

                # Add pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                chunk_metadata_list = query.all()
                logger.debug(f"Retrieved {len(chunk_metadata_list)} chunk metadata records")

                return chunk_metadata_list

        except SQLAlchemyError as e:
            logger.error(f"Database error listing chunk metadata: {e}")
            raise