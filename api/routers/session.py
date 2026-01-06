import uuid
from datetime import datetime
from typing import Annotated, Any, Optional, List
from fastapi import APIRouter, Depends, status, HTTPException
from pydantic import BaseModel
from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import ChatMessage, ChatSession
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.data_model.orm_models import User
from framework.register import Register
from api.routers.auth import validate_user_session
from application.account.chat_session import ChatSessionManager
from application.account.chat_message import ChatMessageManager
from application.account.user import Account
from framework.thread_pool import get_thread_pool


class MessageContent(BaseModel):
    content: str


class MessageRequest(BaseModel):
    """兼容 /api/messages 的请求格式（session_id 在请求体中）"""
    session_id: uuid.UUID
    content: str


class ChatMessageResponse(BaseModel):
    """Response model for chat messages"""
    id: uuid.UUID
    session_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    user_type: Optional[int] = None
    content: dict
    source_file_ids: Optional[List[uuid.UUID]] = None
    sources: Optional[List[dict]] = None
    subgraph_data: Optional[dict] = None
    created_at: datetime

    model_config = {"from_attributes": True}

router = APIRouter(prefix="/session", tags=["session"])

registry = Register()

def get_session_handler() -> ChatSessionManager:
    """Lazy loading function to get session handler after initialization."""
    return registry.get_object("chat_session")

def get_message_handler() -> ChatMessageManager:
    """Lazy loading function to get message handler after initialization."""
    return registry.get_object("chat_message")

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    return registry.get_object("account")

async def list_session_messages(
    session_id: uuid.UUID,
):
    """List session messages asynchronously using thread pool."""
    messages = await get_thread_pool().run_blocking(
        get_message_handler().list_messages_by_session,
        session_id
    )
    # Convert SQLAlchemy models to Pydantic models to ensure proper serialization
    return [ChatMessageResponse.model_validate(msg) for msg in messages]

@router.post("")
async def create_session(
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Create a new chat session asynchronously using thread pool."""
    chat_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return await get_thread_pool().run_blocking(
        get_session_handler().create_session,
        current_user.id,
        chat_name
    )


@router.get("")
async def list_sessions(
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """List user sessions asynchronously using thread pool."""
    return await get_thread_pool().run_blocking(
        get_session_handler().list_sessions_by_user,
        current_user.id
    )


@router.post("/{session_id}/messages")
async def create_message(
    session_id: uuid.UUID,
    message_content: MessageContent,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Create a message asynchronously using thread pool."""
    # Validate user has access to the session
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None or not validate_user_session(session, current_user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    messages = await get_thread_pool().run_blocking(
        get_message_handler().create_message,
        ChatMessage(
            session_id=session_id,
            user_id=current_user.id,
            user_type=current_user.type,
            content={"role": "user", "content": message_content.content},
            created_at=datetime.now()
        )
    )
    return messages




@router.get("/messages", response_model=List[ChatMessageResponse])
async def list_messages_by_user(
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = 100,
    offset: int = 0,
):
    """List messages by current user with pagination. Users can only query their own messages."""
    # 用户只能查询自己的消息
    messages = await get_thread_pool().run_blocking(
        get_message_handler().list_messages_by_user,
        current_user.id,
        limit,
        offset,
    )
    return [ChatMessageResponse.model_validate(msg) for msg in messages]


@router.get("/{session_id}/messages", response_model=List[ChatMessageResponse])
async def list_messages(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """List messages asynchronously using thread pool."""
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None or not validate_user_session(session, current_user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return await list_session_messages(session_id)


@router.delete("/{session_id}")
async def delete_session(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Delete a session asynchronously using thread pool."""
    # 验证用户权限
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None:
        # 不存在，返回200但提示不存在
        return {"message": "Session not found"}
    if not validate_user_session(session, current_user):
        # 存在但不是当前用户的，返回401
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    # 删除会话（会级联删除所有关联的消息）
    success = await get_thread_pool().run_blocking(
        get_session_handler().delete_session,
        session_id
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete session")
    
    # 返回明确的消息，中间件会自动包装为标准响应格式
    return {"message": "Session deleted successfully"}
