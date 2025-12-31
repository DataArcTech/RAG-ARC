from typing import Any, Optional, List, Dict
import uuid
import logging
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ...data_model.orm_models import (
    User,
    FileMetadata, FileStatus,
    FilePermission, PermissionReceiverType
)

logger = logging.getLogger(__name__)


class _PostgreSQLFilesMixin:
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
    

