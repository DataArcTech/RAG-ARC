from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from app_registration import registrator
import jwt
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from application.account.user import UserCreate
from encapsulation.data_model.orm_models import User
from app_registration import Register
from api.routers.auth import get_current_user
from api.routers.auth import Token
from application.account.user import Account

router = APIRouter(prefix="/user", tags=["user"])
registrator = Register()
account_handler: Account = registrator.get_object("account")


@router.get("/me")
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
):
    return current_user