from __future__ import annotations

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

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .base import RelationalDB
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
        logger.info("PostgreSQL engine initialized and tables created")

        return engine
    
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
                updates['updated_at'] = datetime.now(tz=datetime.now().astimezone().tzinfo)

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
        owner_id: Optional[uuid.UUID] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[FileMetadata]:
        """
        List file metadata with optional filtering using SQLAlchemy ORM

        Args:
            status: Optional file status filter
            owner_id: Optional owner ID filter (for user isolation)
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of FileMetadata objects
        """
        try:
            with self.SessionMaker() as session:
                query = session.query(FileMetadata)

                # ✅ Add owner_id filter (for user isolation)
                if owner_id:
                    query = query.filter(FileMetadata.owner_id == owner_id)

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
    
    def count_file_metadata(
        self,
        owner_id: uuid.UUID | None = None,
        status: FileStatus | None = None,
    ) -> int:
        """
        Count file metadata with optional filtering using SQLAlchemy ORM

        Args:
            owner_id: Owner ID filter (for user isolation)
            status: File status filter

        Returns:
            Total count of file metadata records matching the criteria
        """
        try:
            with self.SessionMaker() as session:
                query = session.query(FileMetadata)

                # Add owner_id filter (for user isolation)
                if owner_id:
                    query = query.filter(FileMetadata.owner_id == owner_id)

                # Add status filter
                if status:
                    query = query.filter(FileMetadata.status == status.value)

                count = query.count()
                return count

        except SQLAlchemyError as e:
            logger.error(f"Database error counting file metadata: {e}")
            raise
    
    def list_accessible_files(
        self,
        user_id: uuid.UUID,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[FileMetadata]:
        """
        List all files accessible to a user (owned files + files with permissions).

        Args:
            user_id: UUID of the user
            status: Optional file status filter
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of FileMetadata objects that the user can access (includes owned files and files with permissions)
        """
        try:
            with self.SessionMaker() as session:
                # Get user information
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    logger.warning(f"User not found: {user_id}")
                    return []

                # Build query for files accessible to the user
                # Use UNION to combine:
                # 1. Files owned by the user
                # 2. Files with direct user permission
                # 3. Files with department permission (if user has department)
                # 4. Files with ALL permission

                from sqlalchemy import or_, and_

                # Base query for owned files
                owned_query = session.query(FileMetadata.file_id).filter(
                    FileMetadata.owner_id == user_id
                )

                # Query for files with direct user permission
                user_permission_query = session.query(FilePermission.file_id).filter(
                    and_(
                        FilePermission.user_id == user_id,
                        FilePermission.permission_receiver_type == PermissionReceiverType.USER
                    )
                )

                # Query for files with department permission (if user has department)
                dept_permission_query = None
                if user.department_id:
                    dept_permission_query = session.query(FilePermission.file_id).filter(
                        and_(
                            FilePermission.department_id == user.department_id,
                            FilePermission.permission_receiver_type == PermissionReceiverType.DEPARTMENT
                        )
                    )

                # Query for files with ALL permission
                all_permission_query = session.query(FilePermission.file_id).filter(
                    FilePermission.permission_receiver_type == PermissionReceiverType.ALL
                )

                # Combine all queries using UNION
                file_id_queries = [owned_query, user_permission_query, all_permission_query]
                if dept_permission_query:
                    file_id_queries.append(dept_permission_query)

                # Union all subqueries
                from sqlalchemy import union_all
                # Create union of all file_id queries - use .statement to get Select object
                union_query = union_all(*[q.statement for q in file_id_queries])
                file_id_subquery = union_query.alias('accessible_file_ids')

                # Query FileMetadata for the accessible file IDs
                query = session.query(FileMetadata).filter(
                    FileMetadata.file_id.in_(
                        session.query(file_id_subquery.c.file_id).distinct()
                    )
                )

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
                logger.debug(f"Retrieved {len(file_metadata_list)} accessible files for user {user_id}")

                return file_metadata_list

        except SQLAlchemyError as e:
            logger.error(f"Database error listing accessible files for user {user_id}: {e}")
            raise
    
    def count_accessible_files(
        self,
        user_id: uuid.UUID,
        status: FileStatus | None = None,
    ) -> int:
        """
        Count all files accessible to a user (owned files + files with permissions).

        Args:
            user_id: UUID of the user
            status: Optional file status filter

        Returns:
            Total count of files accessible to the user
        """
        try:
            with self.SessionMaker() as session:
                # Get user information
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    logger.warning(f"User not found: {user_id}")
                    return 0

                from sqlalchemy import or_, and_, union_all

                # Base query for owned files
                owned_query = session.query(FileMetadata.file_id).filter(
                    FileMetadata.owner_id == user_id
                )

                # Query for files with direct user permission
                user_permission_query = session.query(FilePermission.file_id).filter(
                    and_(
                        FilePermission.user_id == user_id,
                        FilePermission.permission_receiver_type == PermissionReceiverType.USER
                    )
                )

                # Query for files with department permission (if user has department)
                dept_permission_query = None
                if user.department_id:
                    dept_permission_query = session.query(FilePermission.file_id).filter(
                        and_(
                            FilePermission.department_id == user.department_id,
                            FilePermission.permission_receiver_type == PermissionReceiverType.DEPARTMENT
                        )
                    )

                # Query for files with ALL permission
                all_permission_query = session.query(FilePermission.file_id).filter(
                    FilePermission.permission_receiver_type == PermissionReceiverType.ALL
                )

                # Combine all queries using UNION
                file_id_queries = [owned_query, user_permission_query, all_permission_query]
                if dept_permission_query:
                    file_id_queries.append(dept_permission_query)

                # Union all subqueries
                # Create union of all file_id queries - use .statement to get Select object
                union_query = union_all(*[q.statement for q in file_id_queries])
                file_id_subquery = union_query.alias('accessible_file_ids')

                # Count FileMetadata for the accessible file IDs
                query = session.query(FileMetadata).filter(
                    FileMetadata.file_id.in_(
                        session.query(file_id_subquery.c.file_id).distinct()
                    )
                )

                # Add status filter
                if status:
                    query = query.filter(FileMetadata.status == status.value)

                count = query.count()
                logger.debug(f"Counted {count} accessible files for user {user_id}")
                return count

        except SQLAlchemyError as e:
            logger.error(f"Database error counting accessible files for user {user_id}: {e}")
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
                updates['updated_at'] = datetime.now(tz=datetime.now().astimezone().tzinfo)

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

    # ==================== USER MANAGEMENT ====================

    def store_user(self, user: User, **kwargs: Any) -> uuid.UUID:
        """Store user metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                session.add(user)
                session.commit()
                logger.debug(f"Stored user: {user.id}")
                return user.id

        except IntegrityError as e:
            logger.error(f"Integrity error storing user (duplicate username?): {e}")
            raise ValueError(f"User with username '{user.user_name}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing user: {e}")
            raise

    def get_user(self, user_id: uuid.UUID, **kwargs: Any) -> Optional[User]:
        """Get user by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                user = session.query(User).filter_by(id=user_id).first()
                if user:
                    # Detach from session to avoid lazy loading issues
                    session.expunge(user)
                return user

        except SQLAlchemyError as e:
            logger.error(f"Database error getting user {user_id}: {e}")
            raise

    def get_user_by_username(self, user_name: str, **kwargs: Any) -> Optional[User]:
        """Get user by username using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                user = session.query(User).filter_by(user_name=user_name).first()
                if user:
                    # Detach from session to avoid lazy loading issues
                    session.expunge(user)
                return user

        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by username {user_name}: {e}")
            raise

    def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[User]:
        """List all users with pagination using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                query = session.query(User).order_by(User.created_at.desc())

                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                users = query.all()
                # Detach from session
                for user in users:
                    session.expunge(user)

                logger.debug(f"Retrieved {len(users)} users")
                return users

        except SQLAlchemyError as e:
            logger.error(f"Database error listing users: {e}")
            raise

    def update_user(self, user_id: uuid.UUID, updates: dict, **kwargs: Any) -> bool:
        """Update user metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                rows_updated = session.query(User).filter_by(id=user_id).update(updates)
                session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated user: {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False

        except IntegrityError as e:
            logger.error(f"Integrity error updating user (duplicate username?): {e}")
            raise ValueError(f"Username already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating user {user_id}: {e}")
            raise

    def delete_user(self, user_id: uuid.UUID, **kwargs: Any) -> bool:
        """Delete user using SQLAlchemy ORM (cascades to sessions and messages)"""
        try:
            with self.SessionMaker() as session:
                rows_deleted = session.query(User).filter_by(id=user_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted user: {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting user {user_id}: {e}")
            raise

    # ==================== CHAT SESSION MANAGEMENT ====================

    def store_chat_session(self, session: ChatSession, **kwargs: Any) -> str:
        """Store chat session metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                db_session.add(session)
                db_session.commit()
                logger.debug(f"Stored chat session: {session.id}")
                return str(session.id)

        except IntegrityError as e:
            logger.error(f"Integrity error storing chat session (invalid user_id?): {e}")
            raise ValueError(f"Invalid user_id or constraint violation")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing chat session: {e}")
            raise

    def get_chat_session(self, session_id: uuid.UUID, **kwargs: Any) -> Optional[ChatSession]:
        """Get chat session by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                chat_session = db_session.query(ChatSession).filter_by(id=session_id).first()
                if chat_session:
                    # Detach from session to avoid lazy loading issues
                    db_session.expunge(chat_session)
                return chat_session

        except SQLAlchemyError as e:
            logger.error(f"Database error getting chat session {session_id}: {e}")
            raise

    def list_chat_sessions_by_user(
        self,
        user_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[ChatSession]:
        """List all chat sessions for a specific user using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                query = db_session.query(ChatSession).filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc())

                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                sessions = query.all()
                # Detach from session
                for s in sessions:
                    db_session.expunge(s)

                logger.debug(f"Retrieved {len(sessions)} chat sessions for user {user_id}")
                return sessions

        except SQLAlchemyError as e:
            logger.error(f"Database error listing chat sessions for user {user_id}: {e}")
            raise

    def get_user_session_count(self, user_id: uuid.UUID, **kwargs: Any) -> int:
        """Get the number of sessions for a user using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                return db_session.query(ChatSession).filter_by(user_id=user_id).count()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user session count for user {user_id}: {e}")
            return 0

    def update_chat_session(self, session_id: uuid.UUID, updates: dict, **kwargs: Any) -> bool:
        """Update chat session metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                rows_updated = db_session.query(ChatSession).filter_by(id=session_id).update(updates)
                db_session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated chat session: {session_id}")
                    return True
                else:
                    logger.warning(f"No chat session found with ID: {session_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating chat session {session_id}: {e}")
            raise

    def delete_chat_session(self, session_id: uuid.UUID, **kwargs: Any) -> bool:
        """Delete chat session using SQLAlchemy ORM (cascades to messages)"""
        try:
            with self.SessionMaker() as db_session:
                rows_deleted = db_session.query(ChatSession).filter_by(id=session_id).delete()
                db_session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted chat session: {session_id}")
                    return True
                else:
                    logger.warning(f"No chat session found with ID: {session_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chat session {session_id}: {e}")
            raise

    # ==================== CHAT MESSAGE MANAGEMENT ====================

    def store_chat_message(self, message: ChatMessage, **kwargs: Any) -> Optional[ChatMessage]:
        """Store chat message metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                db_session.add(message)
                db_session.commit()
                # Expunge the object to avoid lazy loading issues when accessing attributes
                # This prevents SQLAlchemy from trying to reload the object from the database
                # which could fail if the database schema doesn't match the model exactly
                message_id = message.id
                db_session.expunge(message)
                logger.debug(f"Stored chat message: {message_id}")
                return message

        except IntegrityError as e:
            logger.error(f"Integrity error storing chat message (invalid session_id?): {e}")
            raise ValueError(f"Invalid session_id or constraint violation")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing chat message: {e}")
            raise

    def get_chat_message(self, message_id: uuid.UUID, **kwargs: Any) -> Optional[ChatMessage]:
        """Get chat message by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                message = db_session.query(ChatMessage).filter_by(id=message_id).first()
                if message:
                    # Detach from session to avoid lazy loading issues
                    db_session.expunge(message)
                return message

        except SQLAlchemyError as e:
            logger.error(f"Database error getting chat message {message_id}: {e}")
            raise

    def list_chat_messages_by_session(
        self,
        session_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any
    ) -> List[ChatMessage]:
        """List all chat messages for a specific session using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                query = db_session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc())

                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                messages = query.all()
                # Detach from session
                for msg in messages:
                    db_session.expunge(msg)

                logger.debug(f"Retrieved {len(messages)} chat messages for session {session_id}")
                return messages

        except SQLAlchemyError as e:
            logger.error(f"Database error listing chat messages for session {session_id}: {e}")
            raise

    def delete_chat_message(self, message_id: uuid.UUID, **kwargs: Any) -> bool:
        """Delete chat message using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                rows_deleted = db_session.query(ChatMessage).filter_by(id=message_id).delete()
                db_session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted chat message: {message_id}")
                    return True
                else:
                    logger.warning(f"No chat message found with ID: {message_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chat message {message_id}: {e}")
            raise

    def delete_chat_messages_by_session(self, session_id: uuid.UUID, **kwargs: Any) -> int:
        """Delete all chat messages for a specific session using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as db_session:
                rows_deleted = db_session.query(ChatMessage).filter_by(session_id=session_id).delete()
                db_session.commit()

                logger.info(f"Deleted {rows_deleted} chat messages for session {session_id}")
                return rows_deleted

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chat messages for session {session_id}: {e}")
            raise

    # ==================== FILE PERMISSION MANAGEMENT ====================

    def grant_file_permission(
        self,
        file_id: str,
        receiver_type: PermissionReceiverType,
        permission_type: PermissionType,
        granted_by: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        department_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """
        Grant file permission to a user, department, or all users.

        Args:
            file_id: File ID to grant permission for
            receiver_type: Type of receiver (USER, DEPARTMENT, or ALL)
            permission_type: Type of permission (VIEW or EDIT)
            granted_by: User ID who is granting the permission
            user_id: User ID if receiver_type is USER
            department_id: Department ID if receiver_type is DEPARTMENT

        Returns:
            Permission ID (UUID) of the created permission

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        try:
            with self.SessionMaker() as session:
                # Validate file exists
                file_metadata = session.query(FileMetadata).filter_by(file_id=file_id).first()
                if not file_metadata:
                    raise ValueError(f"File not found: {file_id}")

                # Validate receiver type and required IDs
                if receiver_type == PermissionReceiverType.USER:
                    if not user_id:
                        raise ValueError("user_id is required when receiver_type is USER")
                    # Validate user exists
                    user = session.query(User).filter_by(id=user_id).first()
                    if not user:
                        raise ValueError(f"User not found: {user_id}")
                    department_id = None
                elif receiver_type == PermissionReceiverType.DEPARTMENT:
                    if not department_id:
                        raise ValueError("department_id is required when receiver_type is DEPARTMENT")
                    # Validate department exists
                    department = session.query(Department).filter_by(id=department_id).first()
                    if not department:
                        raise ValueError(f"Department not found: {department_id}")
                    user_id = None
                elif receiver_type == PermissionReceiverType.ALL:
                    user_id = None
                    department_id = None

                # Check if permission already exists
                permission = session.query(FilePermission).filter_by(
                    file_id=file_id,
                    permission_receiver_type=receiver_type,
                    user_id=user_id,
                    department_id=department_id
                ).first()

                if permission:
                    # Permission already exists, return existing permission ID
                    permission_id = permission.id
                    logger.info(f"Permission already exists: {permission_id} for file {file_id}")
                else:
                    # Create new permission
                    permission = FilePermission(
                        file_id=file_id,
                        permission_receiver_type=receiver_type,
                        permission_type=permission_type,
                        user_id=user_id,
                        department_id=department_id,
                        granted_by=granted_by
                    )
                    session.add(permission)
                    session.flush()
                    permission_id = permission.id
                    logger.info(f"Created new permission {permission_id} for file {file_id}")

                session.commit()
                return permission_id

        except ValueError:
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error granting file permission: {e}")
            raise ValueError(f"Failed to grant permission: {str(e)}")
        except SQLAlchemyError as e:
            logger.error(f"Database error granting file permission: {e}")
            raise

    def get_file_permission(
        self,
        permission_id: uuid.UUID,
    ) -> Optional[FilePermission]:
        """
        Get a file permission by permission ID.

        Args:
            permission_id: Permission ID to retrieve

        Returns:
            FilePermission object if found, None otherwise
        """
        try:
            with self.SessionMaker() as session:
                permission = session.query(FilePermission).filter_by(id=permission_id).first()
                if permission:
                    session.expunge(permission)
                return permission

        except SQLAlchemyError as e:
            logger.error(f"Database error getting permission {permission_id}: {e}")
            raise

    def revoke_file_permission(
        self,
        permission_id: uuid.UUID,
    ) -> bool:
        """
        Revoke a file permission by permission ID.

        Args:
            permission_id: Permission ID to revoke

        Returns:
            True if permission was revoked, False if not found
        """
        try:
            with self.SessionMaker() as session:
                rows_deleted = session.query(FilePermission).filter_by(id=permission_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.info(f"Revoked permission {permission_id}")
                    return True
                else:
                    logger.warning(f"Permission not found: {permission_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error revoking permission {permission_id}: {e}")
            raise

    def list_file_permissions(
        self,
        file_id: str,
    ) -> List[FilePermission]:
        """
        List all permissions for a specific file.

        Args:
            file_id: File ID to list permissions for
            **kwargs: Additional arguments

        Returns:
            List of FilePermission objects
        """
        try:
            with self.SessionMaker() as session:
                # Eager load user and department relationships
                permissions = session.query(FilePermission).options(
                    joinedload(FilePermission.user).joinedload(User.department),
                    joinedload(FilePermission.department)
                ).filter_by(file_id=file_id).all()
                # Detach from session
                for perm in permissions:
                    session.expunge(perm)
                logger.debug(f"Retrieved {len(permissions)} permissions for file {file_id}")
                return permissions

        except SQLAlchemyError as e:
            logger.error(f"Database error listing permissions for file {file_id}: {e}")
            raise

    def list_user_permissions(
        self,
        user_id: uuid.UUID,
    ) -> List[FilePermission]:
        """
        List all permissions granted to a specific user (direct grants and department grants).

        Args:
            user_id: User ID to list permissions for
            **kwargs: Additional arguments

        Returns:
            List of FilePermission objects
        """
        try:
            with self.SessionMaker() as session:
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    raise ValueError(f"User not found: {user_id}")

                # Get direct user permissions
                user_permissions = session.query(FilePermission).filter(
                    FilePermission.user_id == user_id,
                    FilePermission.permission_receiver_type == PermissionReceiverType.USER
                ).all()

                # Get department permissions if user has a department
                dept_permissions = []
                if user.department_id:
                    dept_permissions = session.query(FilePermission).filter(
                        FilePermission.department_id == user.department_id,
                        FilePermission.permission_receiver_type == PermissionReceiverType.DEPARTMENT
                    ).all()

                # Get ALL permissions
                all_permissions = session.query(FilePermission).filter(
                    FilePermission.permission_receiver_type == PermissionReceiverType.ALL
                ).all()

                # Combine all permissions
                all_perms = user_permissions + dept_permissions + all_permissions

                # Detach from session
                for perm in all_perms:
                    session.expunge(perm)

                logger.debug(f"Retrieved {len(all_perms)} permissions for user {user_id}")
                return all_perms

        except ValueError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error listing permissions for user {user_id}: {e}")
            raise

    def check_file_access(
        self,
        file_id: str,
        user_id: uuid.UUID,
    ) -> Optional[PermissionType]:
        """
        Check if a user has access to a file and return the permission type.

        Args:
            file_id: File ID to check
            user_id: User ID to check access for

        Returns:
            PermissionType (VIEW or EDIT) if user has access, None otherwise
        """
        try:
            with self.SessionMaker() as session:
                # Check if user is the owner
                file_metadata = session.query(FileMetadata).filter_by(file_id=file_id).first()
                if file_metadata and file_metadata.owner_id == user_id:
                    # Owner has full access (EDIT)
                    return PermissionType.EDIT

                # Get user information
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    return None

                # Check for explicit permissions
                # 1. Direct user permission
                user_permission = session.query(FilePermission).filter(
                    FilePermission.file_id == file_id,
                    FilePermission.user_id == user_id,
                    FilePermission.permission_receiver_type == PermissionReceiverType.USER
                ).first()

                if user_permission:
                    return user_permission.permission_type

                # 2. Department permission
                if user.department_id:
                    dept_perm = session.query(FilePermission).filter(
                        FilePermission.file_id == file_id,
                        FilePermission.department_id == user.department_id,
                        FilePermission.permission_receiver_type == PermissionReceiverType.DEPARTMENT
                    ).first()

                    if dept_perm:
                        return dept_perm.permission_type

                # 3. ALL permission
                all_perm = session.query(FilePermission).filter(
                    FilePermission.file_id == file_id,
                    FilePermission.permission_receiver_type == PermissionReceiverType.ALL
                ).first()

                if all_perm:
                    return all_perm.permission_type

                return None

        except SQLAlchemyError as e:
            logger.error(f"Database error checking file access: {e}")
            raise

    def update_file_permission(
        self,
        permission_id: uuid.UUID,
        permission_type: PermissionType,
    ) -> bool:
        """
        Update an existing file permission.

        Args:
            permission_id: Permission ID to update
            permission_type: New permission type (VIEW or EDIT)

        Returns:
            True if permission was updated, False if not found
        """
        try:
            with self.SessionMaker() as session:
                permission = session.query(FilePermission).filter_by(id=permission_id).first()
                if not permission:
                    logger.warning(f"Permission not found: {permission_id}")
                    return False

                permission.permission_type = permission_type

                session.commit()
                logger.info(f"Updated permission {permission_id}")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error updating permission {permission_id}: {e}")
            raise