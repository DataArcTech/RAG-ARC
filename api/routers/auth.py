from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import secrets
from typing import Annotated, Any, Mapping, Optional

import jwt
import redis
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
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from encapsulation.data_model.orm_models import ChatSession, User
from app_registration import Register, initialize as app_initialize
from application.account.user import Account
from config.application.account_config import AccountConfig
from api.schemas.response import StandardResponse
from core.user_management.user import StorageOperationError


def _get_jwt_secret_key() -> str:
    """Resolve JWT signing key.

    Priority:
    1) Explicit env var `JWT_SECRET_KEY`
    2) Persisted key under `RAGARC_RUNTIME_DIR` (or `./local/runtime`) to keep tokens stable across restarts
    3) Generate a new random key and persist it
    """

    value = (os.getenv("JWT_SECRET_KEY") or "").strip()
    if value:
        return value

    # Prefer IOManager-backed persistence (works for both LocalDB and MinIO backends).
    try:
        import app_registration

        io_manager = app_registration.registrator.get_object("io_manager")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"io_manager is required for JWT secret persistence: {exc}") from exc

    if io_manager is None:
        raise RuntimeError("io_manager is required for JWT secret persistence")

    runtime_dir_virtual = str(os.getenv("RAGARC_RUNTIME_DIR", "io://runtime") or "").strip() or "io://runtime"
    if not runtime_dir_virtual.startswith("io://"):
        raise RuntimeError("RAGARC_RUNTIME_DIR must be an io:// virtual path")
    secret_ref = runtime_dir_virtual.rstrip("/") + "/jwt_secret_key"

    try:
        existing = io_manager.get_text_path(secret_ref)
        if existing and existing.strip():
            return existing.strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read persisted JWT secret key at {secret_ref}") from exc

    generated = secrets.token_hex(32)
    try:
        io_manager.put_text(namespace="runtime", key="jwt_secret_key", text=generated)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to persist JWT secret key at {secret_ref}") from exc
    return generated


SECRET_KEY = _get_jwt_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # 24 hours, convenient for debugging.

# Redis configuration (for JWT blacklist)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
JWT_BLACKLIST_PREFIX = "jwt:blacklist:"


def get_redis_client():
    """Get a Redis client for JWT blacklist (best-effort)."""
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD if REDIS_PASSWORD else None,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        # Probe connectivity (blacklist is optional)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis connection failed, JWT blacklist disabled: {e}")
        return None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

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
    type: int  # 0=livingKB / 1=chatKB


class TokenData(BaseModel):
    username: str | None = None


class LoginRequest(BaseModel):
    user_name: str  # Username (for login)
    password: str
    type: Optional[int] = 0  # 0=livingKB / 1=chatKB


class UserCreate(BaseModel):
    name: str
    user_name: str
    password: str
    type: Optional[int] = 0  # 0=livingKB / 1=chatKB


class UserResponse(BaseModel):
    """User response model for API responses"""
    id: str
    user_name: str
    name: str
    status: str
    type: int

    @classmethod
    def from_user(cls, user: User | Mapping[str, Any]) -> "UserResponse":
        """Create `UserResponse` from either a `User` ORM instance or a plain mapping."""

        if isinstance(user, Mapping):
            user_id = user.get("id")
            user_name = str(user.get("user_name") or "")
            name = str(user.get("name") or user_name)
            raw_status = user.get("status")
            if hasattr(raw_status, "value"):
                status_value = str(getattr(raw_status, "value"))
            elif isinstance(raw_status, str):
                status_value = raw_status
            else:
                status_value = "ACTIVE"
            user_type = user.get("type")
            try:
                type_value = int(user_type) if user_type is not None else 0
            except (TypeError, ValueError):
                type_value = 0

            return cls(
                id=str(user_id),
                user_name=user_name,
                name=name,
                status=status_value,
                type=type_value,
            )

        return cls(
            id=str(user.id),
            user_name=user.user_name,
            name=user.name or user.user_name,
            status=user.status.value,
            type=user.type,
        )


class LoginResponse(BaseModel):
    """Login response payload."""
    access_token: str
    token_type: str
    expires_in: int  # Token TTL in seconds
    user: UserResponse


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
        # Best-effort: reject tokens that were explicitly logged out (Redis blacklist).
        try:
            redis_client = get_redis_client()
            if redis_client:
                blacklist_key = f"{JWT_BLACKLIST_PREFIX}{token}"
                exists = redis_client.exists(blacklist_key)
                if exists:
                    logger.info(f"Token found in blacklist, rejecting: {blacklist_key[:50]}...")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User login expired. Please login again.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            else:
                logger.debug("Redis client not available, skipping blacklist check")
        except HTTPException:
            # Re-raise (token is blacklisted).
            raise
        except Exception as e:
            # Redis errors should not block authentication; log and continue.
            logger.warning(f"Redis blacklist check failed: {e}")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format. Please login again.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User login expired. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Catch other JWT decode errors
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Use async version to avoid blocking the event loop
    user = await get_user_async(username=token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_user_optional(
    token: Annotated[str | None, Depends(oauth2_scheme_optional)] = None,
) -> User | None:
    """Get current user from JWT token (optional - returns None if no token provided)"""
    if token is None:
        return None
    try:
        # Best-effort: reject tokens that were explicitly logged out (Redis blacklist).
        try:
            redis_client = get_redis_client()
            if redis_client:
                blacklist_key = f"{JWT_BLACKLIST_PREFIX}{token}"
                exists = redis_client.exists(blacklist_key)
                if exists:
                    logger.info(f"Token found in blacklist, rejecting: {blacklist_key[:50]}...")
                    return None
            else:
                logger.debug("Redis client not available, skipping blacklist check")
        except Exception as e:
            # Redis errors should not block authentication; log and continue.
            logger.warning(f"Redis blacklist check failed: {e}")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        token_data = TokenData(username=username)
    except (InvalidTokenError, Exception):
        # If token is invalid, return None instead of raising exception
        return None
    # Use async version to avoid blocking the event loop
    user = await get_user_async(username=token_data.username)
    return user

@router.post("/token")
async def login_for_access_token_endpoint(request: Request):
    """Login endpoint.

    Supports two input/output modes for backward compatibility:
    - OAuth2 form (`application/x-www-form-urlencoded`): returns `{"access_token": ..., "token_type": ...}`.
    - JSON body: returns `StandardResponse[LoginResponse]`.
    """

    content_type = (request.headers.get("content-type") or "").lower()
    is_form = "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type
    account_handler = get_account_handler()

    if is_form:
        form = await request.form()
        username = str(form.get("username") or "")
        password = str(form.get("password") or "")
        if not username or not password:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing username/password")

        try:
            user = await account_handler.authenticate_user_async(username, password)
        except TypeError:
            user = await account_handler.authenticate_user_async(username, password)  # type: ignore[misc]

        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

        user_type = getattr(user, "type", 0)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.user_name, "type": user_type},
            expires_delta=access_token_expires,
        )
        return Token(access_token=access_token, token_type="bearer", type=int(user_type or 0))

    payload = await request.json()
    username = str(payload.get("user_name") or payload.get("username") or "")
    password = str(payload.get("password") or "")
    login_type = payload.get("type", 0)
    try:
        login_type = int(login_type) if login_type is not None else 0
    except (TypeError, ValueError):
        login_type = 0

    if not username or not password:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing user_name/password")

    try:
        user = await account_handler.authenticate_user_async(username, password, type=login_type)
    except TypeError:
        user = await account_handler.authenticate_user_async(username, password)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    update_login_time = getattr(account_handler, "update_user_login_time_async", None)
    if update_login_time is not None:
        try:
            await update_login_time(user.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update user login time: %s", exc)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_name, "type": getattr(user, "type", login_type)},
        expires_delta=access_token_expires,
    )

    return StandardResponse(
        code=200,
        message="登录成功",
        data=LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.from_user(user),
        ),
    )


@router.post("/register", status_code=status.HTTP_200_OK)
async def register(user: UserCreate) -> StandardResponse[UserResponse]:
    try:
        # Use async registration to avoid blocking the event loop
        new_user = await get_account_handler().register_user_async(user)
        # Convert to response format (request_id is added by middleware).
        return StandardResponse(
            code=200,
            message="success",
            data=UserResponse.from_user(new_user)
        )
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
    except StorageOperationError as e:
        # Handle storage errors, especially duplicate username errors
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

@router.post("/logout")
async def logout(
    request: Request
) -> StandardResponse[None]:
    """Logout endpoint.

    Behavior:
    - Always returns HTTP 200 as long as the server is alive.
    - When Redis is available, the token is added to a blacklist until expiry.
    - When the token is missing/invalid/expired, returns 200 with an explanatory message.
    """

    # Get token from Authorization header.
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        # No token provided.
        return StandardResponse(
            code=200,
            message="认证已失效，无需重复退出",
            data=None
        )
    
    token = authorization[7:]  # Strip "Bearer " prefix.
    
    # Check blacklist (already logged out).
    try:
        redis_client = get_redis_client()
        if redis_client:
            blacklist_key = f"{JWT_BLACKLIST_PREFIX}{token}"
            if redis_client.exists(blacklist_key):
                # Token already blacklisted.
                return StandardResponse(
                    code=200,
                    message="认证已失效，无需重复退出",
                    data=None
                )
    except Exception as e:
        logger.warning(f"Redis blacklist check failed: {e}")
    
    # Decode token (skip exp verification to allow expired tokens to return a stable response).
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        username = payload.get("sub")
        exp = payload.get("exp")
        
        if not username:
            # Invalid token format.
            return StandardResponse(
                code=200,
                message="认证已失效，无需重复退出",
                data=None
            )
        
        # Check if token already expired.
        current_time = datetime.now(timezone.utc).timestamp()
        if exp and exp < current_time:
            # Token expired.
            return StandardResponse(
                code=200,
                message="认证已失效，无需重复退出",
                data=None
            )
        
        # Token still valid; add to blacklist.
        if exp:
            remaining_ttl = int(exp - current_time)
            if remaining_ttl > 0:
                redis_client = get_redis_client()
                if redis_client:
                    try:
                        blacklist_key = f"{JWT_BLACKLIST_PREFIX}{token}"
                        redis_client.setex(blacklist_key, remaining_ttl, "1")
                        logger.info(f"Token blacklisted for user {username}, TTL: {remaining_ttl}s")
                    except Exception as e:
                        logger.error(f"Failed to set token in blacklist: {e}")
                else:
                    logger.warning("Redis client not available, token not blacklisted")
        
        logger.info(f"User {username} logged out")
        return StandardResponse(
            code=200,
            message="退出成功",
            data=None
        )
        
    except InvalidTokenError:
        # Invalid token (bad format/signature/etc.).
        return StandardResponse(
            code=200,
            message="认证已失效，无需重复退出",
            data=None
        )
    except Exception as e:
        # Unexpected error: log but still return a stable response.
        logger.warning(f"Error in logout: {e}")
        return StandardResponse(
            code=200,
            message="退出成功",
            data=None
        )


def validate_user_session(session: ChatSession, current_user: User):
    if session is None:
        logger.warning(f"Session validation failed for user {current_user.id}")
        return False
    if session.user_id != current_user.id:
        logger.warning(f"Session validation failed for session {session.id} and user {current_user.id}")
        return False
    logger.info(f"Validating session {session.id} for user {current_user.id}")
    return True
