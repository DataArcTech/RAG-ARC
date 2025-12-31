from typing import Any, Optional, List, Dict, TYPE_CHECKING
import uuid
import logging

from sqlalchemy.orm import joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ...data_model.orm_models import (
    User, UserStatus,
    Department,
    FileMetadata, FileStatus,
    FilePermission, PermissionReceiverType, PermissionType,
)

logger = logging.getLogger(__name__)


class _PostgreSQLPermissionsMixin:
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

