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


class MessageContent(BaseModel):
    content: str
router = APIRouter(prefix="/session", tags=["session"])

registry = Register()
session_handler: ChatSessionManager = registry.get_object("chat_session")
message_handler: ChatMessageManager = registry.get_object("chat_message")
account_handler: Account = registry.get_object("account")

def list_session_messages(
    session_id: uuid.UUID,
):
    return message_handler.list_messages_by_session(session_id)

@router.post("")
async def create_session(
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    chat_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return session_handler.create_session(current_user.id, chat_name)


@router.get("")
async def list_sessions(
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    return session_handler.list_sessions_by_user(current_user.id)


@router.post("/{session_id}/messages")
async def create_message(
    session_id: uuid.UUID,
    message_content: MessageContent,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    # Validate user has access to the session
    session = session_handler.get_session(session_id)
    if session is None or not validate_user_session(session, current_user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    messages = message_handler.create_message(ChatMessage(session_id=session_id, content={"role": "user", "content": message_content.content}, created_at=datetime.now()))
    return messages


@router.get("/{session_id}/messages")
async def list_messages(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    session = session_handler.get_session(session_id)
    if session is None or not validate_user_session(session, current_user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return list_session_messages(session_id)
