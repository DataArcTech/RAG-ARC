from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Annotated, Optional

import jwt
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    status,
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from encapsulation.data_model.orm_models import ChatSession, User
from app_registration import Register, initialize as app_initialize
from application.account.user import Account
from config.application.account_config import AccountConfig
from pathlib import Path

# Get secret key from environment variable, fallback to default for development
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "f33efd136032819f6017e92272c14afc941eca4fbb94ca266b1d8fa5d8d91107")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # 24小时，方便调试

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

registrator = Register()

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    try:
        return registrator.get_object("account")
    except KeyError:
        # If account is not registered, try to register it now
        logger.warning("Account handler not found, attempting to register...")
        try:
            BASE_DIR = Path(__file__).resolve().parent.parent
            account_config_path = BASE_DIR / "config" / "json_configs" / "account.json"
            if account_config_path.exists():
                # Try to register with detailed error handling
                try:
                    registrator.register(str(account_config_path), "account", AccountConfig)
                    logger.info("Successfully registered account handler")
                    return registrator.get_object("account")
                except Exception as reg_error:
                    # If marshal error, it might be a database connection issue
                    error_msg = str(reg_error)
                    if "marshal" in error_msg.lower():
                        logger.error(f"Marshal error during account registration. This might be a database connection issue. Error: {reg_error}", exc_info=True)
                        # Try to clear any cached connections and retry
                        import gc
                        gc.collect()
                        # Try one more time
                        try:
                            registrator.register(str(account_config_path), "account", AccountConfig)
                            logger.info("Successfully registered account handler after retry")
                            return registrator.get_object("account")
                        except Exception as retry_error:
                            logger.error(f"Retry also failed: {retry_error}", exc_info=True)
                            raise RuntimeError(f"Account handler registration failed due to marshal/database error: {retry_error}") from retry_error
                    else:
                        raise
            else:
                # If config file doesn't exist, try re-initializing
                logger.warning("Account config file not found, attempting full re-initialization...")
                app_initialize()
                return registrator.get_object("account")
        except Exception as e:
            logger.error(f"Failed to register account handler: {e}", exc_info=True)
            raise RuntimeError(f"Account handler is not available: {e}") from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/auth", tags=["auth"])

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class UserCreate(BaseModel):
    name: str
    user_name: str
    password: str


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(username: str):
    """Synchronous version - deprecated, use get_user_async instead"""
    user = get_account_handler().get_user_by_username(username=username)
    return user


async def get_user_async(username: str):
    """Async version that uses thread pool to avoid blocking"""
    return await get_account_handler().get_user_by_username_async(username=username)


async def authenticate_user_async(username: str, password: str):
    """Async version that uses thread pool"""
    user = await get_user_async(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def authenticate_user(username: str, password: str):
    """Synchronous version - deprecated, use authenticate_user_async instead"""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=60)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user_from_token_async(token: str):
    """Get current user from JWT token (async version)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = await get_user_async(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


def get_current_user_from_token(token: str):
    """Get current user from JWT token (synchronous version - deprecated)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_current_user_from_cookie(request: Request):
    """Get current user from cookie."""
    cookies = request.cookies
    auth_token = cookies.get("auth_token")
    
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication token found in cookies",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def login_for_access_token(username: str, password: str) -> Token:
    """Login and return access token."""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_name}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)]):
    """Get current user from JWT token - uses thread pool to avoid blocking"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    # Use async version to avoid blocking the event loop
    user = await get_user_async(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def ws_get_current_user(
    websocket: WebSocket
):
    auth_token = websocket.cookies.get("auth_token")
    if not auth_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None

    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_data = TokenData(username=username)
    except InvalidTokenError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None

    # Use async method to avoid blocking the event loop
    user = await get_user_async(username=token_data.username)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None

    return user

@router.post("/token")
async def login_for_access_token_endpoint(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    # Use async authentication to avoid blocking the event loop
    user = await get_account_handler().authenticate_user_async(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_name}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    try:
        # Use async registration to avoid blocking the event loop
        new_user = await get_account_handler().register_user_async(user)
        return new_user
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")

def validate_user_session(session: ChatSession, current_user: User):
    if session is None:
        logger.warning(f"Session validation failed for user {current_user.id}")
        return False
    if session.user_id != current_user.id:
        logger.warning(f"Session validation failed for session {session.id} and user {current_user.id}")
        return False
    logger.info(f"Validating session {session.id} for user {current_user.id}")
    return True

