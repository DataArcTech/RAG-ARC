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
    name: Optional[str] = None
    type: int = 0
    status: str
    department_id: Optional[uuid.UUID]
    role_id: Optional[uuid.UUID]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login_at: Optional[datetime]


@router.get("/me", response_model=UserMeResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserMeResponse:
    """Return the authenticated user."""

    raw_status = getattr(current_user, "status", None)
    status_value = getattr(raw_status, "value", None)
    if not isinstance(status_value, str) or not status_value:
        status_value = str(raw_status) if raw_status is not None else "ACTIVE"

    return UserMeResponse(
        id=getattr(current_user, "id"),
        user_name=getattr(current_user, "user_name"),
        name=getattr(current_user, "name", None),
        type=int(getattr(current_user, "type", 0) or 0),
        status=status_value,
        department_id=getattr(current_user, "department_id", None),
        role_id=getattr(current_user, "role_id", None),
        created_at=getattr(current_user, "created_at", None),
        updated_at=getattr(current_user, "updated_at", None),
        last_login_at=getattr(current_user, "last_login_at", None),
    )
