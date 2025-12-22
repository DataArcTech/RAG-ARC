from datetime import datetime
import json
from typing import Annotated, Any, Dict, List, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
    Query,
)
from pydantic import BaseModel
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
from framework.thread_pool import get_thread_pool
import uuid
import logging
from core.utils.owner_guard import is_admin_owner, get_admin_owner_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


router = APIRouter(prefix="/rag_inference", tags=["rag_inference"])

registrator = Register()

def get_session_handler() -> ChatSessionManager:
    """Lazy loading function to get session handler after initialization."""
    return registrator.get_object("chat_session")

def get_message_handler() -> ChatMessageManager:
    """Lazy loading function to get message handler after initialization."""
    return registrator.get_object("chat_message")

def get_rag_inference_handler() -> RAGInference:
    """Lazy loading function to get rag inference handler after initialization."""
    return registrator.get_object("rag_inference")

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    return registrator.get_object("account")

manager = ConnectionManager()

class ChatRequest(BaseModel):
    query: str
    return_subgraph: bool = False  # Optional parameter to request subgraph data
    target_owner_id: uuid.UUID | None = None  # Admin-only override
    include_all_owners: bool = False  # Admin-only flag for global retrieval


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    chunks: list | None = None
    subgraph: dict | None = None  # Subgraph visualization data (only if requested)


class GraphOverviewResponse(BaseModel):
    """Response payload for the admin graph overview endpoint."""
    chunks: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# This currently only supports one round of chat, will support multiple rounds once user login is supported.
@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
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
    effective_owner_id: uuid.UUID | None = current_user.id

    if request.include_all_owners:
        if not is_admin_owner(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can access all owners"
            )
        admin_owner = get_admin_owner_id()
        if admin_owner is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ADMIN_OWNER_ID is not configured"
            )
        try:
            effective_owner_id = uuid.UUID(admin_owner)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ADMIN_OWNER_ID must be a valid UUID"
            ) from exc
    elif request.target_owner_id:
        if not is_admin_owner(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can override owner scope"
            )
        effective_owner_id = request.target_owner_id

    response: str = ""
    chunks: list[Chunk] = []
    subgraph_data: GraphData = None
    # Use async version to avoid blocking the event loop
    response, chunks, subgraph_data = await get_rag_inference_handler().chat_async(
        request.query,
        owner_id=effective_owner_id,
        return_subgraph=request.return_subgraph
    )
    return ChatResponse(response=response, chunks=chunks, subgraph=subgraph_data)


@router.get("/graph_overview", response_model=GraphOverviewResponse, status_code=status.HTTP_200_OK)
async def graph_overview(
    current_user: Annotated[User | None, Depends(get_current_user)],
    include_all_owners: bool = Query(
        default=True,
        description="Return the union of all owners when true (admin only).",
    ),
    target_owner_id: Optional[uuid.UUID] = Query(
        default=None,
        description="Specific owner scope when include_all_owners is false.",
    ),
    max_nodes: int = Query(default=1000, ge=10, le=5000),
    max_edges: int = Query(default=5000, ge=10, le=20000),
    include_node_types: Optional[List[str]] = Query(default=None),
):
    """Admin-only endpoint to export a graph overview for visualization."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if not is_admin_owner(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can export graph overviews"
        )

    owner_scope: Optional[uuid.UUID] = None
    if not include_all_owners:
        if target_owner_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="target_owner_id is required when include_all_owners is false"
            )
        owner_scope = target_owner_id

    # Use thread pool to avoid blocking the event loop
    overview = await get_thread_pool().run_blocking(
        get_rag_inference_handler().export_graph_overview,
        owner_id=owner_scope,
        max_nodes=max_nodes,
        max_edges=max_edges,
        include_node_types=include_node_types,
    )
    return GraphOverviewResponse(**overview)


@router.websocket("/stream_chat/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(ws_get_current_user)],
):
    # Accept the connection first - we need to do this before we can close it properly
    await manager.connect(websocket)
    
    # 获取或生成 request_id（从 WebSocket headers 获取，如果没有则生成）
    request_id = websocket.headers.get("X-Request-ID") or str(uuid.uuid4())

    if current_user is None:
        logger.warning(f"WebSocket denied for unauthenticated user on session {session_id}")
        await manager.disconnect(websocket, status.WS_1008_POLICY_VIOLATION)
        return
        
    logger.info(f"WebSocket connection attempt for session_id {session_id} by user {current_user.id}")

    # Validate session ownership at the start (use thread pool to avoid blocking)
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )

    if session is None or not validate_user_session(session, current_user):
        logger.warning(f"Session validation failed for session {session_id} and user {current_user.id}")
        await manager.disconnect(websocket, status.WS_1008_POLICY_VIOLATION)
        return

    try:
        while True:
            # Receive message as text first, then try to parse as JSON
            message_text = await websocket.receive_text()

            # Try to parse as JSON for new format with additional parameters
            override_all = False
            try:
                message_data = json.loads(message_text)
                target_owner_override = None
                if isinstance(message_data, dict):
                    user_message_text = message_data.get("query", message_data.get("content", ""))
                    return_subgraph = message_data.get("return_subgraph", False)
                    target_owner = message_data.get("target_owner_id")
                    include_all_owners = bool(message_data.get("include_all_owners"))
                    if target_owner:
                        try:
                            target_owner_override = uuid.UUID(str(target_owner))
                        except ValueError:
                            await manager.disconnect(websocket, status.WS_1007_INVALID_FRAME_PAYLOAD_DATA)
                            return
                    else:
                        target_owner_override = None
                    if include_all_owners:
                        if not is_admin_owner(current_user.id):
                            await manager.disconnect(websocket, status.WS_1008_POLICY_VIOLATION)
                            return
                        target_owner_override = None
                        override_all = True
                    else:
                        override_all = False
                else:
                    # If JSON parsed but not a dict, treat as plain text
                    user_message_text = message_text
                    return_subgraph = False
                    target_owner_override = None
                    override_all = False
            except (json.JSONDecodeError, ValueError):
                # Not JSON, treat as plain text (backward compatibility)
                user_message_text = message_text
                return_subgraph = False
                target_owner_override = None
                override_all = False

            logger.info(f"Received user message: {user_message_text} (session_id={session_id}, user={getattr(current_user, 'id', None)}, return_subgraph={return_subgraph})")

            user_message = ChatMessage(
                session_id=session_id,
                content={"role": "user", "content": user_message_text},
                created_at=datetime.now()
            )

            # Handle user message creation (use thread pool to avoid blocking)
            user_message = await get_thread_pool().run_blocking(
                get_message_handler().create_message,
                user_message
            )

            # Fetch complete conversation history for multi-round chat (use thread pool to avoid blocking)
            history_messages = await get_thread_pool().run_blocking(
                get_message_handler().list_messages_by_session,
                session_id
            )
            logger.info(f"Conversation history fetched ({len(history_messages)} messages) for session {session_id}")

            # Run RAG inference and create assistant message
            # Convert list of ChatMessage objects to string (e.g., concatenate messages for context)
            history_text = "\n".join(
                f"{msg.content['role']}: {msg.content['content']}" for msg in history_messages
            )
            effective_owner: uuid.UUID | None = current_user.id
            if target_owner_override:
                if not is_admin_owner(current_user.id):
                    await manager.disconnect(websocket, status.WS_1008_POLICY_VIOLATION)
                    return
                effective_owner = target_owner_override
            if override_all:
                admin_owner = get_admin_owner_id()
                if admin_owner is None:
                    await manager.disconnect(websocket, status.WS_1011_INTERNAL_ERROR)
                    return
                try:
                    effective_owner = uuid.UUID(admin_owner)
                except ValueError:
                    await manager.disconnect(websocket, status.WS_1011_INTERNAL_ERROR)
                    return

            # Use async version to avoid blocking the event loop
            assistant_response, chunks, subgraph_data = await get_rag_inference_handler().chat_async(
                history_text,
                effective_owner,
                return_subgraph=return_subgraph
            )
            logger.info(f"Assistant response generated: {assistant_response} (session_id={session_id})")
            
            assistant_message = ChatMessage(
                session_id=session_id, 
                content={"role": "assistant", "content": assistant_response},
                source_file_ids=[chunk.id for chunk in chunks] if chunks else None,
                subgraph_data=subgraph_data if subgraph_data else None,
                created_at=datetime.now()
            )

            # Use thread pool to avoid blocking
            assistant_message = await get_thread_pool().run_blocking(
                get_message_handler().create_message,
                assistant_message
            )
            logger.info(f"Assistant message created: {assistant_message.id}")
            # Send the assistant response back to the client (统一格式包装)
            await manager.send_response(assistant_message, chunks, websocket, subgraph=subgraph_data, request_id=request_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocketDisconnect for session {session_id} and user {getattr(current_user, 'id', None)}")
        await manager.disconnect(websocket)
