import uuid
from datetime import datetime
from typing import Annotated, Any
from fastapi import APIRouter, Depends, WebSocket, status, HTTPException
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
    return await get_thread_pool().run_blocking(
        get_message_handler().list_messages_by_session,
        session_id
    )

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
    
    messages = await get_thread_pool().run_blocking(
        get_message_handler().create_message,
        ChatMessage(session_id=session_id, content={"role": "user", "content": message_content.content}, created_at=datetime.now())
    )
    return messages


@router.get("/{session_id}/messages")
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
