from datetime import datetime
import json
from typing import Annotated
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel
from typing import Optional
from api.routers.auth import get_current_user, ws_get_current_user
from api.routers.connection_manager import ConnectionManager
from api.routers.auth import validate_user_session
from encapsulation.data_model.orm_models import ChatMessage, User
from encapsulation.data_model.schema import Chunk, GraphData
from framework.register import Register
from encapsulation.data_model.orm_models import ChatMessage
from application.rag_inference.module import RAGInference
from application.account.chat_message import ChatMessageManager
from application.account.chat_session import ChatSessionManager
from application.account.user import Account
import uuid
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


router = APIRouter(prefix="/rag_inference", tags=["rag_inference"])

registrator = Register()
session_handler: ChatSessionManager = registrator.get_object("chat_session")
message_handler: ChatMessageManager = registrator.get_object("chat_message")
rag_inference_handler: RAGInference = registrator.get_object("rag_inference")
account_handler: Account = registrator.get_object("account")

manager = ConnectionManager()

class ChatRequest(BaseModel):
    query: str
    return_subgraph: bool = False  # Optional parameter to request subgraph data


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    chunks: list | None = None
    subgraph: dict | None = None  # Subgraph visualization data (only if requested)


# This currently only supports one round of chat, will support multiple rounds once user login is supported.
@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat(
    request: ChatRequest,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Chat endpoint with optional user isolation and subgraph visualization

    Args:
        request: ChatRequest containing query and optional return_subgraph flag

    Returns:
        ChatResponse with LLM response and optional subgraph data
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    response: str = ""
    chunks: list[Chunk] = []
    subgraph_data: GraphData = None
    response, chunks, subgraph_data = rag_inference_handler.chat(
        request.query,
        owner_id=current_user.id,
        return_subgraph=request.return_subgraph
    )
    return ChatResponse(response=response, chunks=chunks, subgraph=subgraph_data)



@router.websocket("/stream_chat/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(ws_get_current_user)],
):
    # Accept the connection first - we need to do this before we can close it properly
    await manager.connect(websocket)

    if current_user is None:
        logger.warning(f"WebSocket denied for unauthenticated user on session {session_id}")
        await manager.disconnect(websocket, status.WS_1008_POLICY_VIOLATION)
        return
        
    logger.info(f"WebSocket connection attempt for session_id {session_id} by user {current_user.id}")

    # Validate session ownership at the start
    session = session_handler.get_session(session_id)

    if session is None or not validate_user_session(session, current_user):
        logger.warning(f"Session validation failed for session {session_id} and user {current_user.id}")
        await manager.disconnect(websocket, status.WS_1008_POLICY_VIOLATION)
        return

    try:
        while True:
            # Receive message as text first, then try to parse as JSON
            message_text = await websocket.receive_text()

            # Try to parse as JSON for new format with additional parameters
            try:
                message_data = json.loads(message_text)
                if isinstance(message_data, dict):
                    user_message_text = message_data.get("query", message_data.get("content", ""))
                    return_subgraph = message_data.get("return_subgraph", False)
                else:
                    # If JSON parsed but not a dict, treat as plain text
                    user_message_text = message_text
                    return_subgraph = False
            except (json.JSONDecodeError, ValueError):
                # Not JSON, treat as plain text (backward compatibility)
                user_message_text = message_text
                return_subgraph = False

            logger.info(f"Received user message: {user_message_text} (session_id={session_id}, user={getattr(current_user, 'id', None)}, return_subgraph={return_subgraph})")

            user_message = ChatMessage(
                session_id=session_id,
                content={"role": "user", "content": user_message_text},
                created_at=datetime.now()
            )

            # Handle user message creation
            user_message = message_handler.create_message(user_message)

            # Fetch complete conversation history for multi-round chat
            history_messages = message_handler.list_messages_by_session(session_id)
            logger.info(f"Conversation history fetched ({len(history_messages)} messages) for session {session_id}")

            # Run RAG inference and create assistant message
            # Convert list of ChatMessage objects to string (e.g., concatenate messages for context)
            history_text = "\n".join(
                f"{msg.content['role']}: {msg.content['content']}" for msg in history_messages
            )
            assistant_response, chunks, subgraph_data = rag_inference_handler.chat(
                history_text,
                current_user.id,
                return_subgraph=return_subgraph
            )
            logger.info(f"Assistant response generated: {assistant_response} (session_id={session_id})")
            assistant_message = ChatMessage(session_id=session_id, content={"role": "assistant", "content": assistant_response}, created_at=datetime.now())
            assistant_message = message_handler.create_message(assistant_message)
            logger.info(f"Assistant message created: {assistant_message.id}")
            # Send the assistant response back to the client
            await manager.send_response(assistant_message, chunks, websocket, subgraph=subgraph_data)

    except WebSocketDisconnect:
        logger.info(f"WebSocketDisconnect for session {session_id} and user {getattr(current_user, 'id', None)}")
        await manager.disconnect(websocket)
