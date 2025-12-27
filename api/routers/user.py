from typing import Annotated

from fastapi import APIRouter, Depends
from encapsulation.data_model.orm_models import User
from api.routers.auth import get_current_user, UserResponse
from app_registration import Register
from application.account.user import Account

router = APIRouter(prefix="/user", tags=["user"])
registrator = Register()

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    return registrator.get_object("account")


@router.get("/me")
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Get current user information"""
    return UserResponse.from_user(current_user)