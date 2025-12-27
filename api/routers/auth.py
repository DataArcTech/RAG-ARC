from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import secrets
from typing import Annotated, Optional

import jwt
import redis
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
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from encapsulation.data_model.orm_models import ChatSession, User
from app_registration import Register, initialize as app_initialize
from application.account.user import Account
from config.application.account_config import AccountConfig
from api.schemas.response import StandardResponse


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

    runtime_dir = Path(os.getenv("RAGARC_RUNTIME_DIR") or "./local/runtime")
    secret_path = runtime_dir / "jwt_secret_key"

    try:
        if secret_path.exists():
            existing = secret_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
    except Exception:  # noqa: BLE001
        pass

    runtime_dir.mkdir(parents=True, exist_ok=True)
    generated = secrets.token_hex(32)
    try:
        secret_path.write_text(generated, encoding="utf-8")
    except Exception:  # noqa: BLE001
        # If we cannot persist, still return a valid secret (tokens won't survive restart).
        pass
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
    user_name: str  # 用户名（登录用）
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
    def from_user(cls, user: User) -> "UserResponse":
        """Create UserResponse from User ORM model"""
        return cls(
            id=str(user.id),
            user_name=user.user_name,
            name=user.name or user.user_name,  # 如果没有name，使用user_name作为默认值
            status=user.status.value,
            type=user.type
        )


class LoginResponse(BaseModel):
    """登录响应数据"""
    access_token: str
    token_type: str
    expires_in: int  # token过期时间（秒）
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
        # 先检查 token 是否在黑名单中
        try:
            redis_client = get_redis_client()
            if redis_client:
                blacklist_key = f"{JWT_BLACKLIST_PREFIX}{token}"
                exists = redis_client.exists(blacklist_key)
                if exists:
                    logger.info(f"Token found in blacklist, rejecting: {blacklist_key[:50]}...")
                    raise credentials_exception
            else:
                logger.debug("Redis client not available, skipping blacklist check")
        except HTTPException:
            # 重新抛出 HTTP 异常（token 在黑名单中）
            raise
        except Exception as e:
            # Redis 操作失败不影响 token 验证，只记录警告
            logger.warning(f"Redis blacklist check failed: {e}")
        
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
    login_data: LoginRequest,
) -> StandardResponse[LoginResponse]:
    """用户登录接口
    
    设计说明：
    - 无论成功或失败都返回 200 状态码
    - 用户名/密码错误时，返回 200，但 data 为 None，message 包含错误信息
    - 401 状态码仅用于未认证访问需要认证的接口
    """
    # 如果 type 为 None，使用默认值 0
    login_type = login_data.type if login_data.type is not None else 0
    # Use async authentication to avoid blocking the event loop, 传入 type 参数
    user = await get_account_handler().authenticate_user_async(login_data.user_name, login_data.password, type=login_type)
    if not user:
        # 用户名/密码错误时，返回 200，但 data 为 None
        return StandardResponse(
            code=200,
            message="用户名或密码错误",
            data=None
        )
    # 更新用户登录时间
    account_handler = get_account_handler()
    await account_handler.update_user_login_time_async(user.id)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_name, "type": user.type}, expires_delta=access_token_expires
    )

    return StandardResponse(
        code=200,
        message="登录成功",
        data=LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # 转换为秒
            user=UserResponse.from_user(user)
        )
    )


@router.post("/register", status_code=status.HTTP_200_OK)
async def register(user: UserCreate) -> StandardResponse[UserResponse]:
    try:
        # Use async registration to avoid blocking the event loop
        new_user = await get_account_handler().register_user_async(user)
        # 转换为响应格式，code 设置为 200（中间件会自动添加 request_id）
        return StandardResponse(
            code=200,
            message="success",
            data=UserResponse.from_user(new_user)
        )
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")

@router.post("/logout")
async def logout(
    request: Request
) -> StandardResponse[None]:
    """用户退出接口 - 将 token 加入 Redis 黑名单
    
    设计说明：
    - 只要服务端不挂，始终返回 200 状态码
    - 如果 token 无效或已过期，返回 200，但 message 提示认证失效
    - 如果 token 有效，正常处理登出逻辑
    """
    # 从请求头获取 token
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        # 没有提供 token，返回认证失效提示
        return StandardResponse(
            code=200,
            message="认证已失效，无需重复退出",
            data=None
        )
    
    token = authorization[7:]  # 移除 "Bearer " 前缀
    
    # 检查 token 是否在黑名单中（已登出）
    try:
        redis_client = get_redis_client()
        if redis_client:
            blacklist_key = f"{JWT_BLACKLIST_PREFIX}{token}"
            if redis_client.exists(blacklist_key):
                # token 已在黑名单中，说明已经登出过
                return StandardResponse(
                    code=200,
                    message="认证已失效，无需重复退出",
                    data=None
                )
    except Exception as e:
        logger.warning(f"Redis blacklist check failed: {e}")
    
    # 尝试解析 token（不验证过期时间，因为可能已过期）
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        username = payload.get("sub")
        exp = payload.get("exp")
        
        if not username:
            # token 格式无效
            return StandardResponse(
                code=200,
                message="认证已失效，无需重复退出",
                data=None
            )
        
        # 检查 token 是否已过期
        current_time = datetime.now(timezone.utc).timestamp()
        if exp and exp < current_time:
            # token 已过期
            return StandardResponse(
                code=200,
                message="认证已失效，无需重复退出",
                data=None
            )
        
        # token 有效，加入黑名单
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
        # token 无效（格式错误、签名错误等）
        return StandardResponse(
            code=200,
            message="认证已失效，无需重复退出",
            data=None
        )
    except Exception as e:
        # 其他异常，记录日志但返回成功
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
