"""Owner ID resolution logic."""
import uuid
import logging
from fastapi import HTTPException, status
from encapsulation.data_model.orm_models import User
from api.routers.rag_inference_handlers import get_default_owner_id
from core.utils.owner_guard import is_admin_owner, get_admin_owner_id

logger = logging.getLogger(__name__)


def resolve_effective_owner(
    current_user: User,
    target_owner_id: uuid.UUID | None,
    include_all_owners: bool
) -> uuid.UUID | None:
    """Resolve effective owner ID based on user permissions and request parameters."""
    effective_owner: uuid.UUID | None = get_default_owner_id(current_user)
    
    if include_all_owners:
        if not is_admin_owner(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can access all owners"
            )
        admin_owner = get_admin_owner_id()
        if admin_owner is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ADMIN_OWNER_ID is not configured"
            )
        try:
            effective_owner = uuid.UUID(admin_owner)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ADMIN_OWNER_ID must be a valid UUID"
            ) from exc
    elif target_owner_id:
        if not is_admin_owner(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can override owner scope"
            )
        effective_owner = target_owner_id
    
    return effective_owner
