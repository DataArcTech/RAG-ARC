import logging
import uuid
from typing import List, Optional

from fastapi import HTTPException

from encapsulation.data_model.orm_models import FilePermission, PermissionReceiverType, PermissionType

logger = logging.getLogger(__name__)


class KnowledgePermissionMixin:
    def get_file_id_by_permission_id(self, permission_id: uuid.UUID) -> Optional[str]:
        """
        Get the file ID by permission ID.

        Args:
            permission_id: Permission ID to look up

        Returns:
            File ID string if permission exists, None otherwise
        """
        permission = self.file_storage.metadata_store.get_file_permission(permission_id)  # type: ignore[attr-defined]
        if not permission or not permission.file_id:
            return None
        return permission.file_id

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
        """
        metadata = self.file_storage.get_file_metadata(file_id)  # type: ignore[attr-defined]
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        if self.check_file_access(file_id, granted_by) != PermissionType.EDIT:
            raise HTTPException(status_code=403, detail="You are not allowed to grant permissions for this file")

        try:
            permission_id = self.file_storage.metadata_store.grant_file_permission(  # type: ignore[attr-defined]
                file_id=file_id,
                receiver_type=receiver_type,
                permission_type=permission_type,
                granted_by=granted_by,
                user_id=user_id,
                department_id=department_id,
            )
            logger.info("Granted %s permission for file %s to %s", permission_type.value, file_id, receiver_type.value)
            return permission_id
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to grant file permission: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to grant permission: {exc}") from exc

    def revoke_file_permission(
        self,
        file_id: str,
        user_id: uuid.UUID,
        receiver_type: PermissionReceiverType | None = None,
        target_user_id: uuid.UUID | None = None,
        department_id: uuid.UUID | None = None,
    ) -> bool:
        """
        Revoke file permission from a user, department, or all users.

        Args:
            file_id: File ID to revoke permission for
            user_id: User ID requesting the revocation (must have EDIT permission)
            receiver_type: Type of receiver to revoke permission from
            target_user_id: Target user ID if receiver_type is USER
            department_id: Department ID if receiver_type is DEPARTMENT

        Returns:
            True if permission was revoked, False if permission not found
        """
        metadata = self.file_storage.get_file_metadata(file_id)  # type: ignore[attr-defined]
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        if self.check_file_access(file_id, user_id) != PermissionType.EDIT:
            raise HTTPException(status_code=403, detail="Only users with EDIT permission can revoke permissions")

        try:
            result = self.file_storage.metadata_store.revoke_file_permission(  # type: ignore[attr-defined]
                file_id=file_id,
                receiver_type=receiver_type,
                user_id=target_user_id,
                department_id=department_id,
            )
            if result:
                logger.info("Revoked permission for file %s", file_id)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to revoke file permission: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to revoke permission: {exc}") from exc

    def list_file_permissions(self, file_id: str, user_id: uuid.UUID) -> List[FilePermission]:
        """
        List all permissions for a file (requires at least VIEW access).

        Args:
            file_id: File ID to list permissions for
            user_id: User ID requesting the list

        Returns:
            List of FilePermission objects
        """
        metadata = self.file_storage.get_file_metadata(file_id)  # type: ignore[attr-defined]
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        permission_type = self.check_file_access(file_id, user_id)
        if permission_type is None:
            raise HTTPException(status_code=403, detail="You are not allowed to list permissions for this file")

        try:
            permissions = self.file_storage.metadata_store.list_file_permissions(file_id)  # type: ignore[attr-defined]
            logger.info("Retrieved %s permissions for file %s", len(permissions), file_id)
            return permissions
        except Exception as exc:
            logger.error("Failed to list file permissions: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to list permissions: {exc}") from exc

    def list_user_permissions(self, user_id: uuid.UUID) -> List[FilePermission]:
        """
        List all permissions granted to a specific user.

        Args:
            user_id: User ID to list permissions for

        Returns:
            List of FilePermission objects
        """
        try:
            permissions = self.file_storage.metadata_store.list_user_permissions(user_id)  # type: ignore[attr-defined]
            logger.info("Retrieved %s permissions for user %s", len(permissions), user_id)
            return permissions
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to list user permissions: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to list permissions: {exc}") from exc

    def check_file_access(self, file_id: str, user_id: uuid.UUID) -> Optional[PermissionType]:
        """
        Check if a user has access to a file and return the permission type.

        Args:
            file_id: File ID to check
            user_id: User ID to check access for

        Returns:
            PermissionType (VIEW or EDIT) if user has access, None otherwise
        """
        try:
            permission_type = self.file_storage.metadata_store.check_file_access(  # type: ignore[attr-defined]
                file_id=file_id,
                user_id=user_id,
            )
            return permission_type
        except Exception as exc:
            logger.error("Failed to check file access: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to check access: {exc}") from exc

    def update_file_permission(
        self,
        permission_id: uuid.UUID,
        permission_type: PermissionType,
        user_id: uuid.UUID,
    ) -> bool:
        """
        Update an existing file permission.

        Args:
            permission_id: Permission ID to update
            permission_type: New permission type (VIEW or EDIT)
            user_id: User ID requesting the update (must have EDIT permission)

        Returns:
            True if permission was updated, False if not found
        """
        permission = self.file_storage.metadata_store.get_file_permission(permission_id)  # type: ignore[attr-defined]
        if not permission:
            raise HTTPException(status_code=404, detail="Permission not found")

        file_id = permission.file_id
        if self.check_file_access(file_id, user_id) != PermissionType.EDIT:
            raise HTTPException(status_code=403, detail="Only users with EDIT permission can update permissions")

        try:
            result = self.file_storage.metadata_store.update_file_permission(  # type: ignore[attr-defined]
                permission_id=permission_id,
                permission_type=permission_type,
            )
            if result:
                logger.info("Updated permission %s", permission_id)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to update file permission: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to update permission: {exc}") from exc

