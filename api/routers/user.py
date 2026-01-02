from datetime import datetime
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import User

router = APIRouter(prefix="/user", tags=["user"])


class UserMeResponse(BaseModel):
    """Sanitized current-user payload (never includes credentials)."""

    id: uuid.UUID
    user_name: str
    name: Optional[str]
    type: int
    status: str
    department_id: Optional[uuid.UUID]
    role_id: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]


@router.get("/me", response_model=UserMeResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserMeResponse:
    """Return the authenticated user."""

    status_value = getattr(current_user.status, "value", None)
    if not isinstance(status_value, str) or not status_value:
        status_value = str(current_user.status)

    return UserMeResponse(
        id=current_user.id,
        user_name=current_user.user_name,
        name=current_user.name,
        type=current_user.type,
        status=status_value,
        department_id=current_user.department_id,
        role_id=current_user.role_id,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        last_login_at=current_user.last_login_at,
    )

