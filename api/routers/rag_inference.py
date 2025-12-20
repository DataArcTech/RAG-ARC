from datetime import datetime
import json
import asyncio
import os
import threading
from typing import Annotated, Any, Dict, List, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.routers.auth import get_current_user, ws_get_current_user
from api.routers.auth import validate_user_session
from api.sse import (
    delta_envelope,
    iter_text_deltas,
    new_chatcmpl_id,
    now_epoch_seconds,
    openai_chat_completion_chunk,
    sse_done,
    sse_json,
)
from encapsulation.data_model.orm_models import ChatMessage, User
from encapsulation.data_model.schema import Chunk, GraphData
from framework.register import Register
from application.rag_inference.module import RAGInference
from application.account.chat_message import ChatMessageManager
from application.account.chat_session import ChatSessionManager
from application.account.user import Account
from framework.thread_pool import get_thread_pool
import uuid
import logging
from core.utils.owner_guard import is_admin_owner, get_admin_owner_id
from core.presentation.evidence import build_chat_evidence
from config.output_limits import CHAT_TOP_CHUNKS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


router = APIRouter(prefix="/rag_inference", tags=["rag_inference"])

registrator = Register()

session_handler: ChatSessionManager | None = None
message_handler: ChatMessageManager | None = None
rag_inference_handler: RAGInference | None = None


def get_session_handler() -> ChatSessionManager:
    """Lazy loading function to get session handler after initialization."""
    global session_handler
    if session_handler is None:
        session_handler = registrator.get_object("chat_session")
    return session_handler

def get_message_handler() -> ChatMessageManager:
    """Lazy loading function to get message handler after initialization."""
    global message_handler
    if message_handler is None:
        message_handler = registrator.get_object("chat_message")
    return message_handler

def get_rag_inference_handler() -> RAGInference:
    """Lazy loading function to get rag inference handler after initialization."""
    global rag_inference_handler
    if rag_inference_handler is None:
        rag_inference_handler = registrator.get_object("rag_inference")
    return rag_inference_handler

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    return registrator.get_object("account")

class ChatRequest(BaseModel):
    query: str
    return_subgraph: bool = False  # Optional parameter to request subgraph data
    target_owner_id: uuid.UUID | None = None  # Admin-only override
    include_all_owners: bool = False  # Admin-only flag for global retrieval
    include_evidence: bool = False  # Whether to include chunk/seed/triple summary


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    chunks: list | None = None
    subgraph: dict | None = None  # Subgraph visualization data (only if requested)
    evidence: Dict[str, Any] | None = None


class GraphOverviewResponse(BaseModel):
    """Response payload for the admin graph overview endpoint."""
    chunks: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]

def _build_stream_chat_payload(
    message: ChatMessage,
    chunks: list[Chunk],
    subgraph: dict | None = None,
    evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    message_dict = {
        "id": str(message.id),
        "session_id": str(message.session_id),
        "content": message.content,
        "created_at": (message.created_at.isoformat() if message.created_at else None),
    }
    chunks_dict = [
        {
            "id": str(chunk.id),
            "content": chunk.content,
            "metadata": chunk.metadata,
            "graph": chunk.graph.to_dict(),
        }
        for chunk in chunks
    ]
    response_dict: Dict[str, Any] = {"message": message_dict, "chunks": chunks_dict}
    if subgraph is not None:
        response_dict["subgraph"] = subgraph
    if evidence is not None:
        response_dict["evidence"] = evidence
    return response_dict


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

    rag_inference_handler = get_rag_inference_handler()
    response_text: str = ""
    chunks: list[Chunk] = []
    subgraph_data: GraphData = None
    needs_subgraph = request.return_subgraph or request.include_evidence
    response_text, chunks, subgraph_data, subgraph_info = await rag_inference_handler.chat_async(
        request.query,
        owner_id=effective_owner_id,
        return_subgraph=needs_subgraph
    )
    evidence = None
    if request.include_evidence:
        graph_store = None
        try:
            graph_store = rag_inference_handler.get_graph_store()
        except Exception:  # noqa: BLE001
            graph_store = None
        evidence = build_chat_evidence(
            chunks,
            subgraph_data=subgraph_data,
            subgraph_info=subgraph_info,
            max_chunks=CHAT_TOP_CHUNKS,
            graph_store=graph_store,
        )
    subgraph_payload = subgraph_data if request.return_subgraph else None
    return ChatResponse(response=response_text, chunks=chunks, subgraph=subgraph_payload, evidence=evidence)


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


@router.get("/stream_chat/{session_id}")
async def stream_chat_sse(
    session_id: uuid.UUID,
    query: str = Query(..., description="User query text"),
    return_subgraph: bool = Query(default=False),
    target_owner_id: Optional[uuid.UUID] = Query(default=None),
    include_all_owners: bool = Query(default=False),
    include_evidence: bool = Query(default=False),
    current_user: Annotated[User | None, Depends(get_current_user)] = None,
):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    logger.info("SSE stream_chat request for session_id %s by user %s", session_id, current_user.id)

    # Validate session ownership at the start (use thread pool to avoid blocking)
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None or not validate_user_session(session, current_user):
        logger.warning("Session validation failed for session %s and user %s", session_id, current_user.id)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    message_handler = get_message_handler()
    rag_inference_handler = get_rag_inference_handler()

    effective_owner: uuid.UUID | None = current_user.id
    if include_all_owners:
        if not is_admin_owner(current_user.id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin users can access all owners")
        admin_owner = get_admin_owner_id()
        if admin_owner is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ADMIN_OWNER_ID is not configured")
        try:
            effective_owner = uuid.UUID(admin_owner)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ADMIN_OWNER_ID must be a valid UUID") from exc
    elif target_owner_id:
        if not is_admin_owner(current_user.id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin users can override owner scope")
        effective_owner = target_owner_id

    model_name = os.getenv("CHAT_MODEL_NAME") or os.getenv("OPENAI_CHAT_MODEL") or "rag-arc"

    async def event_generator():
        chunk_id = new_chatcmpl_id()
        created = now_epoch_seconds()

        # Qwen/OpenAI-compatible streams typically start with a chunk that sets role=assistant.
        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(role="assistant", content=""),
            )
        )

        user_message = ChatMessage(
            session_id=session_id,
            content={"role": "user", "content": query},
            created_at=datetime.now(),
        )
        await get_thread_pool().run_blocking(message_handler.create_message, user_message)

        try:
            token_stream, chunks, subgraph_data, subgraph_info = await asyncio.to_thread(
                rag_inference_handler.stream_chat,
                query,
                effective_owner,
                return_subgraph=(return_subgraph or include_evidence),
            )
        except Exception as exc:  # noqa: BLE001
            yield sse_json({"error": {"message": str(exc)}})
            yield sse_done()
            return

        response_parts: list[str] = []
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stream_error: list[Exception | None] = [None]

        def _run_stream() -> None:
            try:
                for chunk in token_stream:
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as exc:  # noqa: BLE001
                stream_error[0] = exc
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_run_stream, daemon=True).start()

        while True:
            piece = await queue.get()
            if piece is None:
                break
            for delta_piece in iter_text_deltas(piece):
                response_parts.append(delta_piece)
                yield sse_json(
                    openai_chat_completion_chunk(
                        chunk_id=chunk_id,
                        model=model_name,
                        created=created,
                        delta=delta_envelope(role=None, content=delta_piece),
                    )
                )
                await asyncio.sleep(0)

        if stream_error[0] is not None:
            yield sse_json({"error": {"message": str(stream_error[0])}})
            yield sse_done()
            return

        assistant_response = "".join(response_parts)
        assistant_message = ChatMessage(
            session_id=session_id,
            content={"role": "assistant", "content": assistant_response},
            source_file_ids=[chunk.id for chunk in chunks] if chunks else None,
            subgraph_data=subgraph_data if return_subgraph else None,
            created_at=datetime.now(),
        )
        assistant_message = await get_thread_pool().run_blocking(
            message_handler.create_message, assistant_message
        )

        # Keep legacy "include_evidence/return_subgraph" capability but transmit it via
        # OpenAI-compatible tool_calls so clients consuming only delta.content are unaffected.
        if include_evidence or return_subgraph:
            evidence = None
            if include_evidence:
                graph_store = None
                try:
                    graph_store = rag_inference_handler.get_graph_store()
                except Exception:  # noqa: BLE001
                    graph_store = None
                evidence = build_chat_evidence(
                    chunks,
                    subgraph_data=subgraph_data,
                    subgraph_info=subgraph_info,
                    max_chunks=CHAT_TOP_CHUNKS,
                    graph_store=graph_store,
                )

            payload = _build_stream_chat_payload(
                assistant_message,
                chunks,
                subgraph=subgraph_data if return_subgraph else None,
                evidence=evidence,
            )

            tool_calls = [
                {
                    "index": 0,
                    "id": f"call_{assistant_message.id}",
                    "type": "function",
                    "function": {
                        "name": "rag_arc_payload",
                        "arguments": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            ]
            yield sse_json(
                openai_chat_completion_chunk(
                    chunk_id=chunk_id,
                    model=model_name,
                    created=created,
                    delta=delta_envelope(role=None, tool_calls=tool_calls),
                )
            )

        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(),
                finish_reason="stop",
            )
        )
        yield sse_done()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@router.websocket("/stream_chat/{session_id}")
async def stream_chat_ws(
    websocket: WebSocket,
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(ws_get_current_user)],
):
    await websocket.accept()
    if current_user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    session = await get_thread_pool().run_blocking(get_session_handler().get_session, session_id)
    if session is None or not validate_user_session(session, current_user):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    message_handler = get_message_handler()
    rag_inference_handler = get_rag_inference_handler()

    try:
        while True:
            message_text = await websocket.receive_text()
            return_subgraph = False
            target_owner_id: uuid.UUID | None = None
            include_all_owners = False
            include_evidence = False
            query = message_text

            try:
                payload = json.loads(message_text)
                if isinstance(payload, dict):
                    query = payload.get("query") or payload.get("message") or query
                    return_subgraph = bool(payload.get("return_subgraph", False))
                    include_all_owners = bool(payload.get("include_all_owners", False))
                    include_evidence = bool(payload.get("include_evidence", False))
                    if payload.get("target_owner_id"):
                        target_owner_id = uuid.UUID(str(payload["target_owner_id"]))
            except Exception:  # noqa: BLE001
                pass

            effective_owner: uuid.UUID | None = current_user.id
            if include_all_owners:
                if not is_admin_owner(current_user.id):
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                admin_owner = get_admin_owner_id()
                if admin_owner is None:
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                    return
                effective_owner = uuid.UUID(admin_owner)
            elif target_owner_id is not None:
                if not is_admin_owner(current_user.id):
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                effective_owner = target_owner_id

            user_message = ChatMessage(
                session_id=session_id,
                content={"role": "user", "content": query},
                created_at=datetime.now(),
            )
            await get_thread_pool().run_blocking(message_handler.create_message, user_message)
            history_messages = await get_thread_pool().run_blocking(
                message_handler.list_messages_by_session,
                session_id,
            )
            history_text = "\n".join(
                f"{msg.content['role']}: {msg.content['content']}" for msg in history_messages
            )

            assistant_response, chunks, subgraph_data, subgraph_info = await rag_inference_handler.chat_async(
                history_text,
                owner_id=effective_owner,
                return_subgraph=(return_subgraph or include_evidence),
            )
            assistant_message = ChatMessage(
                session_id=session_id,
                content={"role": "assistant", "content": assistant_response},
                source_file_ids=[chunk.id for chunk in chunks] if chunks else None,
                subgraph_data=subgraph_data if return_subgraph else None,
                created_at=datetime.now(),
            )
            assistant_message = await get_thread_pool().run_blocking(
                message_handler.create_message, assistant_message
            )

            evidence = None
            if include_evidence:
                graph_store = None
                try:
                    graph_store = rag_inference_handler.get_graph_store()
                except Exception:  # noqa: BLE001
                    graph_store = None
                evidence = build_chat_evidence(
                    chunks,
                    subgraph_data=subgraph_data,
                    subgraph_info=subgraph_info,
                    max_chunks=CHAT_TOP_CHUNKS,
                    graph_store=graph_store,
                )

            response_payload = _build_stream_chat_payload(
                assistant_message,
                chunks,
                subgraph=subgraph_data if return_subgraph else None,
                evidence=evidence,
            )
            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnect (session_id=%s user=%s)", session_id, current_user.id)
