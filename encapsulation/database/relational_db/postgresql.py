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
from functools import cached_property

from sqlalchemy import create_engine, text, Engine, Column, String, BigInteger, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .base import RelationalDB
from .models.file_metadata import FileMetadata, FileStatus
from .models.parsed_content_metadata import ParsedContentMetadata, ParsedContentStatus
from .models.chunks_metadata import ChunksMetadata, ChunksStatus
from framework.shared_module_decorator import shared_module

logger = logging.getLogger(__name__)

Base = declarative_base()


class FileMetadataTable(Base):
    """SQLAlchemy table model for file metadata"""
    __tablename__ = 'file_metadata'
    
    asset_id = Column(String(255), primary_key=True)
    blob_key = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(100))
    checksum = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime, nullable=False)
    updated_at = Column(DateTime, default=datetime, onupdate=datetime, nullable=False)
    original_path = Column(Text)


class ParsedContentMetadataTable(Base):
    """SQLAlchemy table model for parsed content metadata"""
    __tablename__ = 'parsed_content_metadata'
    
    parsed_content_id = Column(String(255), primary_key=True)
    source_asset_id = Column(String(255), nullable=False)
    blob_key = Column(String(500), nullable=False)
    content_size = Column(BigInteger, nullable=False)
    checksum = Column(String(64), nullable=False)
    parser_type = Column(String(100), nullable=False)
    parser_version = Column(String(50), nullable=True)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime, nullable=False)
    updated_at = Column(DateTime, default=datetime, onupdate=datetime, nullable=False)
    content_type = Column(String(100), default="text/markdown", nullable=False)
    parsing_config = Column(Text)
    page_count = Column(BigInteger)
    language = Column(String(10))


class ChunksMetadataTable(Base):
    """SQLAlchemy table model for chunks metadata"""
    __tablename__ = 'chunks_metadata'

    chunks_id = Column(String(255), primary_key=True)
    source_parsed_content_id = Column(String(255), nullable=False)
    blob_key = Column(String(500), nullable=False)
    chunks_count = Column(BigInteger, nullable=False)
    content_size = Column(BigInteger, nullable=False)
    checksum = Column(String(64), nullable=False)
    chunking_strategy = Column(String(100), nullable=False)
    chunking_version = Column(String(50), nullable=True)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime, nullable=False)
    updated_at = Column(DateTime, default=datetime, onupdate=datetime, nullable=False)
    content_type = Column(String(100), default="application/json", nullable=False)
    processing_time_ms = Column(BigInteger)
    chunking_config = Column(Text)


@shared_module
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
    
    @cached_property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine (cached)"""
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
    
    def _row_to_file_metadata(self, row) -> FileMetadata:
        """Convert database row to FileMetadata dataclass"""
        return FileMetadata(
            asset_id=row.asset_id,
            blob_key=row.blob_key,
            filename=row.filename,
            status=FileStatus(row.status),
            file_size=row.file_size,
            content_type=row.content_type,
            checksum=row.checksum,
            created_at=row.created_at,
            updated_at=row.updated_at,
            original_path=row.original_path
        )
    
    def _row_to_parsed_content_metadata(self, row) -> ParsedContentMetadata:
        """Convert database row to ParsedContentMetadata dataclass"""
        return ParsedContentMetadata(
            parsed_content_id=row.parsed_content_id,
            source_asset_id=row.source_asset_id,
            blob_key=row.blob_key,
            content_size=row.content_size,
            checksum=row.checksum,
            parser_type=row.parser_type,
            parser_version=row.parser_version,
            status=ParsedContentStatus(row.status),
            created_at=row.created_at,
            updated_at=row.updated_at,
            content_type=row.content_type,
            parsing_config=row.parsing_config,
            page_count=row.page_count,
            language=row.language
        )

    def _row_to_chunks_metadata(self, row) -> ChunksMetadata:
        """Convert database row to ChunksMetadata dataclass"""
        return ChunksMetadata(
            chunks_id=row.chunks_id,
            source_parsed_content_id=row.source_parsed_content_id,
            blob_key=row.blob_key,
            chunks_count=row.chunks_count,
            content_size=row.content_size,
            checksum=row.checksum,
            chunking_strategy=row.chunking_strategy,
            chunking_version=row.chunking_version,
            status=ChunksStatus(row.status),
            created_at=row.created_at,
            updated_at=row.updated_at,
            content_type=row.content_type,
            processing_time_ms=row.processing_time_ms,
            chunking_config=row.chunking_config
        )

    def store_file_metadata(
        self,
        file_metadata: FileMetadata,
        **kwargs: Any,
    ) -> str:
        """Store file metadata using raw SQL"""
        insert_sql = text("""
        INSERT INTO file_metadata 
        (asset_id, blob_key, filename, status, file_size, content_type, checksum, created_at, updated_at, original_path)
        VALUES (:asset_id, :blob_key, :filename, :status, :file_size, :content_type, :checksum, :created_at, :updated_at, :original_path)
        """)
        
        try:
            with self.engine.connect() as conn:
                conn.execute(insert_sql, {
                    "asset_id": file_metadata.asset_id,
                    "blob_key": file_metadata.blob_key,
                    "filename": file_metadata.filename,
                    "status": file_metadata.status.value,
                    "file_size": file_metadata.file_size,
                    "content_type": file_metadata.content_type,
                    "checksum": file_metadata.checksum,
                    "created_at": file_metadata.created_at,
                    "updated_at": file_metadata.updated_at,
                    "original_path": file_metadata.original_path
                })
                conn.commit()
                logger.debug(f"Stored file metadata for asset: {file_metadata.asset_id}")
                return file_metadata.asset_id
                
        except IntegrityError:
            logger.error(f"File metadata with asset_id '{file_metadata.asset_id}' already exists")
            raise ValueError(f"File metadata with asset_id '{file_metadata.asset_id}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing file metadata: {e}")
            raise
    
    def get_file_metadata(self, asset_id: str, **kwargs: Any) -> Optional[FileMetadata]:
        """Retrieve file metadata by asset ID using raw SQL"""
        select_sql = text("SELECT * FROM file_metadata WHERE asset_id = :asset_id")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(select_sql, {"asset_id": asset_id})
                row = result.fetchone()
                
                if row:
                    logger.debug(f"Retrieved file metadata for asset: {asset_id}")
                    return self._row_to_file_metadata(row)
                
                return None
                
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving file metadata: {e}")
            raise
    
    def update_file_metadata(
        self,
        asset_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update file metadata using raw SQL"""
        if not updates:
            return True
        
        # Add updated_at timestamp with Beijing timezone
        updates['updated_at'] = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        
        # Build dynamic UPDATE query
        set_clauses = ", ".join([f"{key} = :{key}" for key in updates.keys()])
        update_sql = text(f"UPDATE file_metadata SET {set_clauses} WHERE asset_id = :asset_id")
        
        # Convert enum values to strings if needed
        params = updates.copy()
        for key, value in params.items():
            if isinstance(value, FileStatus):
                params[key] = value.value
        params['asset_id'] = asset_id
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(update_sql, params)
                conn.commit()
                
                if result.rowcount > 0:
                    logger.debug(f"Updated file metadata for asset: {asset_id}")
                    return True
                
                logger.warning(f"No file metadata found to update for asset: {asset_id}")
                return False
                
        except SQLAlchemyError as e:
            logger.error(f"Database error updating file metadata: {e}")
            raise
    
    def delete_file_metadata(self, asset_id: str, **kwargs: Any) -> bool:
        """Delete file metadata using raw SQL"""
        delete_sql = text("DELETE FROM file_metadata WHERE asset_id = :asset_id")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(delete_sql, {"asset_id": asset_id})
                conn.commit()
                
                if result.rowcount > 0:
                    logger.debug(f"Deleted file metadata for asset: {asset_id}")
                    return True
                
                logger.warning(f"No file metadata found to delete for asset: {asset_id}")
                return False
                
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting file metadata: {e}")
            raise
    
    def update_file_status(
        self,
        asset_id: str,
        new_status: FileStatus,
        **kwargs: Any,
    ) -> bool:
        """Update file processing status"""
        return self.update_file_metadata(
            asset_id,
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
        """List file metadata with optional filtering using raw SQL"""
        base_sql = "SELECT * FROM file_metadata"
        params = {}
        
        # Add status filter
        if status:
            base_sql += " WHERE status = :status"
            params['status'] = status.value
        
        # Add ordering
        base_sql += " ORDER BY created_at DESC"
        
        # Add pagination
        if limit:
            base_sql += " LIMIT :limit"
            params['limit'] = limit
        if offset:
            base_sql += " OFFSET :offset"
            params['offset'] = offset
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(base_sql), params)
                rows = result.fetchall()
                
                file_metadata_list = [self._row_to_file_metadata(row) for row in rows]
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
        """Store parsed content metadata using raw SQL"""
        insert_sql = text("""
        INSERT INTO parsed_content_metadata 
        (parsed_content_id, source_asset_id, blob_key, content_size, checksum, 
         parser_type, parser_version, status, created_at, updated_at, content_type, 
         parsing_config, page_count, language)
        VALUES (:parsed_content_id, :source_asset_id, :blob_key, :content_size, 
                :checksum, :parser_type, :parser_version, :status, :created_at, :updated_at, 
                :content_type, :parsing_config, :page_count, :language)
        """)
        
        try:
            with self.engine.connect() as conn:
                conn.execute(insert_sql, {
                    "parsed_content_id": parsed_metadata.parsed_content_id,
                    "source_asset_id": parsed_metadata.source_asset_id,
                    "blob_key": parsed_metadata.blob_key,
                    "content_size": parsed_metadata.content_size,
                    "checksum": parsed_metadata.checksum,
                    "parser_type": parsed_metadata.parser_type,
                    "parser_version": parsed_metadata.parser_version,
                    "status": parsed_metadata.status.value,
                    "created_at": parsed_metadata.created_at,
                    "updated_at": parsed_metadata.updated_at,
                    "content_type": parsed_metadata.content_type,
                    "parsing_config": parsed_metadata.parsing_config,
                    "page_count": parsed_metadata.page_count,
                    "language": parsed_metadata.language
                })
                conn.commit()
                logger.debug(f"Stored parsed content metadata: {parsed_metadata.parsed_content_id}")
                return parsed_metadata.parsed_content_id
                
        except IntegrityError:
            logger.error(f"Parsed content metadata with ID '{parsed_metadata.parsed_content_id}' already exists")
            raise ValueError(f"Parsed content metadata with ID '{parsed_metadata.parsed_content_id}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing parsed content metadata: {e}")
            raise
    
    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional[ParsedContentMetadata]:
        """Retrieve parsed content metadata by ID using raw SQL"""
        select_sql = text("SELECT * FROM parsed_content_metadata WHERE parsed_content_id = :parsed_content_id")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(select_sql, {"parsed_content_id": parsed_content_id})
                row = result.fetchone()
                
                if row:
                    logger.debug(f"Retrieved parsed content metadata: {parsed_content_id}")
                    return self._row_to_parsed_content_metadata(row)
                
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
        """Update parsed content metadata using raw SQL"""
        if not updates:
            return True
        
        # Add updated_at timestamp with Beijing timezone
        updates['updated_at'] = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        
        # Build dynamic UPDATE query
        set_clauses = ", ".join([f"{key} = :{key}" for key in updates.keys()])
        update_sql = text(f"UPDATE parsed_content_metadata SET {set_clauses} WHERE parsed_content_id = :parsed_content_id")
        
        # Convert enum values to strings if needed
        params = updates.copy()
        for key, value in params.items():
            if isinstance(value, ParsedContentStatus):
                params[key] = value.value
        params['parsed_content_id'] = parsed_content_id
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(update_sql, params)
                conn.commit()
                
                if result.rowcount > 0:
                    logger.debug(f"Updated parsed content metadata: {parsed_content_id}")
                    return True
                
                logger.warning(f"No parsed content metadata found to update: {parsed_content_id}")
                return False
                
        except SQLAlchemyError as e:
            logger.error(f"Database error updating parsed content metadata: {e}")
            raise
    
    def delete_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Delete parsed content metadata using raw SQL"""
        delete_sql = text("DELETE FROM parsed_content_metadata WHERE parsed_content_id = :parsed_content_id")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(delete_sql, {"parsed_content_id": parsed_content_id})
                conn.commit()
                
                if result.rowcount > 0:
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
        source_asset_id: Optional[str] = None,
        status: Optional[ParsedContentStatus] = None,
        parser_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ParsedContentMetadata]:
        """List parsed content metadata with optional filtering using raw SQL"""
        base_sql = "SELECT * FROM parsed_content_metadata"
        params = {}
        conditions = []
        
        # Add filters
        if source_asset_id:
            conditions.append("source_asset_id = :source_asset_id")
            params['source_asset_id'] = source_asset_id
        if status:
            conditions.append("status = :status")
            params['status'] = status.value
        if parser_type:
            conditions.append("parser_type = :parser_type")
            params['parser_type'] = parser_type
        
        # Add WHERE clause if conditions exist
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)
        
        # Add ordering
        base_sql += " ORDER BY created_at DESC"
        
        # Add pagination
        if limit:
            base_sql += " LIMIT :limit"
            params['limit'] = limit
        if offset:
            base_sql += " OFFSET :offset"
            params['offset'] = offset
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(base_sql), params)
                rows = result.fetchall()
                
                parsed_metadata_list = [self._row_to_parsed_content_metadata(row) for row in rows]
                logger.debug(f"Retrieved {len(parsed_metadata_list)} parsed content metadata records")
                
                return parsed_metadata_list
                
        except SQLAlchemyError as e:
            logger.error(f"Database error listing parsed content metadata: {e}")
            raise

    # ==================== CHUNKS METADATA METHODS ====================

    def store_chunks_metadata(
        self,
        chunks_metadata: ChunksMetadata,
        **kwargs: Any,
    ) -> str:
        """Store chunks metadata in PostgreSQL"""

        sql = """
        INSERT INTO chunks_metadata (
            chunks_id, source_parsed_content_id, blob_key, chunks_count,
            content_size, checksum, chunking_strategy, chunking_version,
            status, created_at, updated_at, content_type,
            processing_time_ms, chunking_config
        ) VALUES (
            :chunks_id, :source_parsed_content_id, :blob_key, :chunks_count,
            :content_size, :checksum, :chunking_strategy, :chunking_version,
            :status, :created_at, :updated_at, :content_type,
            :processing_time_ms, :chunking_config
        )
        """

        params = {
            'chunks_id': chunks_metadata.chunks_id,
            'source_parsed_content_id': chunks_metadata.source_parsed_content_id,
            'blob_key': chunks_metadata.blob_key,
            'chunks_count': chunks_metadata.chunks_count,
            'content_size': chunks_metadata.content_size,
            'checksum': chunks_metadata.checksum,
            'chunking_strategy': chunks_metadata.chunking_strategy,
            'chunking_version': chunks_metadata.chunking_version,
            'status': chunks_metadata.status.value,
            'created_at': chunks_metadata.created_at,
            'updated_at': chunks_metadata.updated_at,
            'content_type': chunks_metadata.content_type,
            'processing_time_ms': chunks_metadata.processing_time_ms,
            'chunking_config': chunks_metadata.chunking_config
        }

        try:
            with self.engine.connect() as conn:
                conn.execute(text(sql), params)
                conn.commit()
                logger.info(f"Stored chunks metadata: {chunks_metadata.chunks_id}")
                return chunks_metadata.chunks_id

        except IntegrityError as e:
            if "already exists" in str(e) or "duplicate key" in str(e):
                raise ValueError(f"Chunks with ID {chunks_metadata.chunks_id} already exists")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error storing chunks metadata: {e}")
            raise

    def get_chunks_metadata(self, chunks_id: str, **kwargs: Any) -> Optional[ChunksMetadata]:
        """Get chunks metadata by chunks_id"""

        sql = """
        SELECT chunks_id, source_parsed_content_id, blob_key, chunks_count,
               content_size, checksum, chunking_strategy, chunking_version,
               status, created_at, updated_at, content_type,
               processing_time_ms, chunking_config
        FROM chunks_metadata
        WHERE chunks_id = :chunks_id
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), {'chunks_id': chunks_id})
                row = result.fetchone()

                if row:
                    return self._row_to_chunks_metadata(row)
                return None

        except SQLAlchemyError as e:
            logger.error(f"Database error getting chunks metadata for {chunks_id}: {e}")
            raise

    def update_chunks_metadata(
        self,
        chunks_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update chunks metadata fields"""

        if not updates:
            return False

        # Always update the updated_at timestamp
        updates['updated_at'] = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

        set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
        sql = f"UPDATE chunks_metadata SET {set_clause} WHERE chunks_id = :chunks_id"

        params = updates.copy()
        params['chunks_id'] = chunks_id

        # Convert enum values to strings if needed
        if 'status' in params and hasattr(params['status'], 'value'):
            params['status'] = params['status'].value

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params)
                conn.commit()

                rows_affected = result.rowcount
                if rows_affected > 0:
                    logger.debug(f"Updated chunks metadata: {chunks_id}")
                    return True
                else:
                    logger.warning(f"No chunks found with ID: {chunks_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating chunks metadata {chunks_id}: {e}")
            raise

    def delete_chunks_metadata(self, chunks_id: str, **kwargs: Any) -> bool:
        """Delete chunks metadata"""

        sql = "DELETE FROM chunks_metadata WHERE chunks_id = :chunks_id"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), {'chunks_id': chunks_id})
                conn.commit()

                rows_affected = result.rowcount
                if rows_affected > 0:
                    logger.info(f"Deleted chunks metadata: {chunks_id}")
                    return True
                else:
                    logger.warning(f"No chunks found with ID: {chunks_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chunks metadata {chunks_id}: {e}")
            raise

    def update_chunks_status(
        self,
        chunks_id: str,
        new_status: ChunksStatus,
        **kwargs: Any,
    ) -> bool:
        """Update chunks processing status"""

        return self.update_chunks_metadata(
            chunks_id,
            {'status': new_status.value},
            **kwargs
        )

    def list_chunks_metadata(
        self,
        source_parsed_content_id: Optional[str] = None,
        status: Optional[ChunksStatus] = None,
        chunking_strategy: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ChunksMetadata]:
        """List chunks metadata with optional filtering"""

        base_sql = """
        SELECT chunks_id, source_parsed_content_id, blob_key, chunks_count,
               content_size, checksum, chunking_strategy, chunking_version,
               status, created_at, updated_at, content_type,
               processing_time_ms, chunking_config
        FROM chunks_metadata
        """

        conditions = []
        params = {}

        # Add filters
        if source_parsed_content_id:
            conditions.append("source_parsed_content_id = :source_parsed_content_id")
            params['source_parsed_content_id'] = source_parsed_content_id
        if status:
            conditions.append("status = :status")
            params['status'] = status.value
        if chunking_strategy:
            conditions.append("chunking_strategy = :chunking_strategy")
            params['chunking_strategy'] = chunking_strategy

        # Add WHERE clause if we have conditions
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)

        # Add ordering
        base_sql += " ORDER BY created_at DESC"

        # Add pagination
        if limit:
            base_sql += " LIMIT :limit"
            params['limit'] = limit
        if offset:
            base_sql += " OFFSET :offset"
            params['offset'] = offset

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(base_sql), params)
                rows = result.fetchall()

                chunks_metadata_list = [self._row_to_chunks_metadata(row) for row in rows]
                logger.debug(f"Retrieved {len(chunks_metadata_list)} chunks metadata records")

                return chunks_metadata_list

        except SQLAlchemyError as e:
            logger.error(f"Database error listing chunks metadata: {e}")
            raise