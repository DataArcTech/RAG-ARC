from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, TYPE_CHECKING
import uuid
import asyncio
from functools import partial

import jwt
from fastapi import HTTPException, Request, status
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from framework.module import AbstractModule
from core.user_management.user import UserValidationError
from framework.thread_pool import get_thread_pool
import logging
if TYPE_CHECKING:
    from config.application.account_config import AccountConfig


logger = logging.getLogger(__name__)

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class UserCreate(BaseModel):
    name: str
    user_name: str
    password: str
    type: int  # 0=livingKB / 1=chatKB


class Account(AbstractModule):
    
    def __init__(self, config: "AccountConfig"):
        super().__init__(config=config)
        self.user_storage = config.user_storage_config.build()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking function in a separate thread to avoid blocking the event loop."""
        return await get_thread_pool().run_blocking(func, *args, **kwargs)

    def get_user_by_id(self, user_id: uuid.UUID):
        return self.user_storage.get_user(user_id)

    def get_user_by_username(self, username: str, type: Optional[int] = None):
        return self.user_storage.get_user_by_username(username, type=type)
    
    async def get_user_by_id_async(self, user_id: uuid.UUID):
        """Async wrapper for get_user_by_id."""
        return await self._run_blocking(self.get_user_by_id, user_id)

    async def get_user_by_username_async(self, username: str, type: Optional[int] = None):
        """Async wrapper for get_user_by_username."""
        return await self._run_blocking(self.get_user_by_username, username, type)
    
    # Authentication utility functions
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)

    def authenticate_user(self, username: str, password: str):
        """Authenticate a user with username and password (synchronous)."""
        user = self.get_user_by_username(username)
        if not user:
            return False
        if not self.verify_password(password, user.hashed_password):
            return False
        return user

    async def authenticate_user_async(self, username: str, password: str, type: Optional[int] = None):
        """Authenticate a user with username, password and type (async, non-blocking)."""
        user = await self.get_user_by_username_async(username, type=type)
        if not user:
            return False
        # Password verification is CPU-bound but fast, can run in executor if needed
        if not self.verify_password(password, user.hashed_password):
            return False
        return user

    def register_user(self, user_data: UserCreate):
        """Register a new user (synchronous)."""
        try:
            # 如果 type 为 None，使用默认值 0
            user_type = user_data.type if user_data.type is not None else 0
            # 检查用户是否已存在（通过 user_name 和 type 组合）
            existing_user = self.user_storage.get_user_by_username(user_data.user_name, type=user_type)
            if existing_user:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists for this type")
            
            hashed_password = self.get_password_hash(user_data.password)
            new_user_id = self.user_storage.create_user(
                user_name=user_data.user_name,
                hashed_password=hashed_password,
                type=user_type,
                name=user_data.name
            )
            # 返回创建的用户对象
            new_user = self.user_storage.get_user(new_user_id)
            return new_user
        except HTTPException:
            raise
        except (IntegrityError, UserValidationError) as e:
            logger.error(f"User creation failed: {str(e)}")
            raise HTTPException(status_code=400, detail="User creation failed")

    async def register_user_async(self, user_data: UserCreate):
        """Register a new user (async, non-blocking)."""
        return await self._run_blocking(self.register_user, user_data)

    def create_session(self, user_id: uuid.UUID):
        return self.user_storage.create_chat_session(user_id)

    async def create_session_async(self, user_id: uuid.UUID):
        """Create a chat session (async, non-blocking)."""
        return await self._run_blocking(self.create_session, user_id)

    def update_user_login_time(self, user_id: uuid.UUID):
        """Update user's last login time (synchronous)."""
        return self.user_storage.update_user(user_id, {"last_login_at": datetime.now()})

    async def update_user_login_time_async(self, user_id: uuid.UUID):
        """Update user's last login time (async, non-blocking)."""
        return await self._run_blocking(self.update_user_login_time, user_id)