from datetime import datetime
import json
import asyncio
import os
import time
import threading
import contextvars
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
    sse_json_wrapped,
)
from api.routers.chatbot import (
    _sanitize_title,
    _fallback_title,
    _generate_title_messages,
    _sse_json,
    _sse_done,
    _build_sources_for_frontend,
    _filter_sources_by_sup_keys,
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


def _get_shared_document_owner_id() -> uuid.UUID:
    """Get shared document owner ID from environment variable for unified file retrieval."""
    raw = os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001")
    try:
        return uuid.UUID(str(raw))
    except ValueError as exc:
        raise RuntimeError("CHATBOT_SHARED_DOCUMENT_OWNER_ID must be a valid UUID") from exc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


router = APIRouter(prefix="/rag_inference", tags=["rag_inference"])


# Title生成函数（复用chatbot.py中的函数）
async def _generate_title_via_llm(
    rag_inference_handler: RAGInference,
    user_text: str,
    assistant_text: str,
) -> str:
    """生成标题，复用chatbot.py中的辅助函数"""
    llm = getattr(rag_inference_handler, "llm", None)
    if llm is None:
        return _fallback_title(user_text)

    messages = _generate_title_messages(user_text, assistant_text)

    def _run():
        return llm.chat(messages)

    try:
        raw = await get_thread_pool().run_blocking(_run)
    except Exception:  # noqa: BLE001
        return _fallback_title(user_text)

    title = _sanitize_title(str(raw or ""))
    return title or _fallback_title(user_text)

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


class StreamChatRequest(BaseModel):
    """Request model for POST SSE stream chat endpoint"""
    query: str
    return_subgraph: bool = False
    target_owner_id: Optional[uuid.UUID] = None
    include_all_owners: bool = False
    include_evidence: bool = False


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
    # Use shared document owner ID for unified file retrieval across all users
    effective_owner_id: uuid.UUID | None = _get_shared_document_owner_id()

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
    response_text, chunks, subgraph_data, subgraph_info, raw_llm_response, raw_mindmap_response = await rag_inference_handler.chat_async(
        request.query,
        owner_id=effective_owner_id,
        return_subgraph=needs_subgraph,
        current_user_query=request.query,
    )
    
    # Log full response details (including graph payload).
    logger.info(
        "Chat response: text_length=%d, chunks_count=%d, subgraph_nodes=%d, subgraph_edges=%d, raw_response=%s, raw_mindmap_response=%s",
        len(response_text) if response_text else 0,
        len(chunks) if chunks else 0,
        len(subgraph_data.get("nodes", [])) if subgraph_data else 0,
        len(subgraph_data.get("edges", [])) if subgraph_data else 0,
        json.dumps(raw_llm_response, ensure_ascii=False, default=str) if raw_llm_response else None,
        raw_mindmap_response if raw_mindmap_response else None
    )
    if subgraph_data:
        logger.debug("Subgraph data: %s", json.dumps(subgraph_data, ensure_ascii=False, default=str))
    
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


@router.post("/stream_chat/{session_id}")
async def stream_chat_sse(
    session_id: uuid.UUID,
    request: StreamChatRequest,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """
    SSE stream chat endpoint with user authentication required (POST method)
    
    Request body (JSON):
    {
        "query": "User query text",
        "return_subgraph": false,
        "target_owner_id": null,
        "include_all_owners": false,
        "include_evidence": false
    }
    
    Args:
        session_id: Chat session ID (path parameter)
        request: StreamChatRequest containing query and optional flags
        current_user: Current authenticated user (required, from JWT token)
    
    Returns:
        StreamingResponse with SSE events (text/event-stream)
    """
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # Extract parameters from request body
    query = request.query
    return_subgraph = request.return_subgraph
    target_owner_id = request.target_owner_id
    include_all_owners = request.include_all_owners
    include_evidence = request.include_evidence

    # Guard: only livingKB users (type=0) may request subgraph/evidence generation.
    if return_subgraph or include_evidence:
        user_type = getattr(current_user, "type", 0)
        if user_type != 0:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only livingKB users (type=0) can request subgraph generation"
            )

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

    # Use shared document owner ID for unified file retrieval across all users
    effective_owner: uuid.UUID | None = _get_shared_document_owner_id()
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
        request_id = uuid.uuid4().hex
        progress_seq = 0

        # Qwen/OpenAI-compatible streams typically start with a chunk that sets role=assistant.
        yield sse_json_wrapped(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(role="assistant", content=""),
            ),
            request_id=request_id
        )

        user_message = ChatMessage(
            session_id=session_id,
            content={"role": "user", "content": query},
            created_at=datetime.now(),
        )
        user_message = await get_thread_pool().run_blocking(message_handler.create_message, user_message)

        # Load recent history messages (aligned with WebSocket + chatbot behavior).
        history_messages = await get_thread_pool().run_blocking(
            message_handler.list_messages_by_session,
            session_id,
        )
        
        # Exclude the message we just created (avoid duplication).
        history_messages = [msg for msg in history_messages if msg.id != user_message.id]
        
        # 判断是否是第一轮对话（检查是否有assistant消息）
        first_turn = not any(
            msg.content.get("role") == "assistant" 
            for msg in history_messages 
            if isinstance(msg.content, dict)
        )
        
        # Limit history length (similar to chatbot.py `_normalize_history`).
        # Default: keep the last 5 turns (10 messages: user + assistant).
        context_turns = int(os.getenv("SSE_CONTEXT_TURNS", "5"))
        max_history_messages = context_turns * 2
        if len(history_messages) > max_history_messages:
            history_messages = history_messages[-max_history_messages:]
        
        # Build history text (similar to WebSocket implementation).
        history_text = "\n".join(
            f"{msg.content['role']}: {msg.content['content']}" for msg in history_messages
        ) if history_messages else None

        # Enforce a rough context budget (similar to chatbot.py `_ensure_context_within_limit`).
        # Estimate tokens for the history text.
        if history_text:
            # Simple token estimate (aligned with chatbot.py `_estimate_tokens`).
            history_tokens = len(history_text) if any(ord(ch) >= 128 for ch in history_text) else len(history_text) // 4
            max_context_tokens = int(os.getenv("SSE_MAX_CONTEXT_TOKENS", "8192"))
            threshold_fraction = float(os.getenv("SSE_MAX_CONTEXT_FRACTION", "0.9"))
            allowed_tokens = int(max_context_tokens * threshold_fraction)
            
            if history_tokens > allowed_tokens:
                logger.warning(
                    "History too long: estimated_tokens=%d, allowed=%d, truncating history",
                    history_tokens,
                    allowed_tokens,
                )
                # If history is too long, reduce the number of retained turns further.
                reduced_turns = max(1, context_turns // 2)
                max_history_messages = reduced_turns * 2
                if len(history_messages) > max_history_messages:
                    history_messages = history_messages[-max_history_messages:]
                    history_text = "\n".join(
                        f"{msg.content['role']}: {msg.content['content']}" for msg in history_messages
                    )

        response_parts: list[str] = []
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stream_error: list[Exception | None] = [None]
        prepared: dict[str, Any] = {}

        def _emit_progress(payload: dict[str, Any]) -> None:
            nonlocal progress_seq
            progress_seq += 1
            envelope = dict(payload or {})
            envelope.setdefault("v", 1)
            envelope.setdefault("type", "progress")
            envelope.setdefault("ts_ms", int(time.time() * 1000))
            envelope.setdefault("request_id", request_id)
            envelope.setdefault("seq", progress_seq)
            asyncio.run_coroutine_threadsafe(queue.put({"kind": "progress", "payload": envelope}), loop)

        def _run_stream() -> None:
            try:
                _emit_progress({"stage": "prepare", "status": "start"})
                try:
                    token_stream, chunks, subgraph_data, subgraph_info = rag_inference_handler.stream_chat(
                        query,
                        effective_owner,
                        return_subgraph=(return_subgraph or include_evidence),
                        progress_callback=_emit_progress,
                        history_text=history_text if history_text else None,
                    )
                except TypeError:
                    # Backward compatibility: older implementations may not accept `history_text`.
                    token_stream, chunks, subgraph_data, subgraph_info = rag_inference_handler.stream_chat(
                        query,
                        effective_owner,
                        return_subgraph=(return_subgraph or include_evidence),
                    )
                prepared["chunks"] = chunks
                prepared["subgraph_data"] = subgraph_data
                prepared["subgraph_info"] = subgraph_info
                prepared["raw_llm_response"] = None  # Streaming does not capture a full raw response.
                prepared["raw_mindmap_response"] = None  # Streaming does not capture a full raw mindmap response.
                _emit_progress({"stage": "prepare", "status": "end"})
                _emit_progress({"stage": "generate", "status": "start"})
                
                # Log token stream collection
                token_count = 0
                total_token_length = 0
                for chunk in token_stream:
                    token_count += 1
                    chunk_str = str(chunk) if chunk else ""
                    total_token_length += len(chunk_str)
                    if token_count <= 5:  # Log first few tokens
                        logger.debug(
                            "SSE token_stream chunk %d: chunk_type=%s chunk_length=%d chunk_preview=%s",
                            token_count,
                            type(chunk).__name__,
                            len(chunk_str),
                            chunk_str[:100] if chunk_str else "None",
                        )
                    asyncio.run_coroutine_threadsafe(queue.put({"kind": "token", "text": chunk}), loop)
                
                logger.info(
                    "SSE token_stream collection completed: total_tokens=%d total_length=%d",
                    token_count,
                    total_token_length,
                )
                _emit_progress({"stage": "generate", "status": "end"})
            except Exception as exc:  # noqa: BLE001
                stream_error[0] = exc
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Capture contextvars (including correlation_id) before creating thread
        ctx = contextvars.copy_context()
        threading.Thread(target=lambda: ctx.run(_run_stream), daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break

            if isinstance(item, dict) and item.get("kind") == "progress":
                tool_calls = [
                    {
                        "index": 0,
                        "id": f"call_progress_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": "rag_arc_progress",
                            "arguments": json.dumps(
                                item.get("payload") or {},
                                ensure_ascii=False,
                                default=str,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ]
                yield sse_json_wrapped(
                    openai_chat_completion_chunk(
                        chunk_id=chunk_id,
                        model=model_name,
                        created=created,
                        delta=delta_envelope(role=None, tool_calls=tool_calls),
                    ),
                    request_id=request_id
                )
                continue

            if isinstance(item, dict) and item.get("kind") == "token":
                piece = str(item.get("text") or "")
                if not piece:
                    continue
                for delta_piece in iter_text_deltas(piece):
                    response_parts.append(delta_piece)
                    yield sse_json_wrapped(
                        openai_chat_completion_chunk(
                            chunk_id=chunk_id,
                            model=model_name,
                            created=created,
                            delta=delta_envelope(role=None, content=delta_piece),
                        ),
                        request_id=request_id
                    )
                    await asyncio.sleep(0)
                continue

        if stream_error[0] is not None:
            yield sse_json_wrapped(
                {"error": {"message": str(stream_error[0])}},
                request_id=request_id,
                code=500,
                message="error"
            )
            yield sse_done()
            return

        assistant_response = "".join(response_parts)
        if not assistant_response:
            logger.warning(
                "SSE assistant_response is empty after streaming; "
                "query=%r owner_id=%s history_len=%d prepared_chunks=%d",
                query,
                str(effective_owner),
                len(history_messages),
                len(prepared.get("chunks") or []),
            )

        chunks = prepared.get("chunks") or []
        subgraph_data = prepared.get("subgraph_data")
        subgraph_info = prepared.get("subgraph_info")
        raw_llm_response = prepared.get("raw_llm_response")
        raw_mindmap_response = prepared.get("raw_mindmap_response")

        # Align with WebSocket behavior: when return_subgraph=True, always generate a mindmap
        # to ensure the graph is tightly coupled to the current chat answer.
        if return_subgraph:
            # Generate the mindmap from the current user query to keep it consistent with the answer.
            try:
                subgraph_data, raw_mindmap_response = await get_thread_pool().run_blocking(
                    rag_inference_handler._generate_mindmap,
                    query,
                    assistant_response,
                    chunks,
                )
                logger.info("SSE generated mindmap: %d nodes, %d edges", 
                           len(subgraph_data.get("nodes", [])) if subgraph_data else 0,
                           len(subgraph_data.get("edges", [])) if subgraph_data else 0)
            except Exception as exc:
                logger.warning("Failed to generate mindmap in SSE: %s", exc)
                # If generation fails, keep the original subgraph_data (if any).
                if not subgraph_data:
                    subgraph_data = None
                    raw_mindmap_response = None

        # Log full response details (including graph payload).
        logger.info(
            "SSE chat response: text_length=%d, chunks_count=%d, subgraph_nodes=%d, subgraph_edges=%d, raw_response=%s, raw_mindmap_response=%s, response_content=%s",
            len(assistant_response) if assistant_response else 0,
            len(chunks) if chunks else 0,
            len(subgraph_data.get("nodes", [])) if subgraph_data else 0,
            len(subgraph_data.get("edges", [])) if subgraph_data else 0,
            json.dumps(raw_llm_response, ensure_ascii=False, default=str) if raw_llm_response else None,
            raw_mindmap_response if raw_mindmap_response else None,
            assistant_response[:500] if assistant_response else None  # Log first 500 chars of response content
        )
        if subgraph_data:
            logger.debug("SSE subgraph data: %s", json.dumps(subgraph_data, ensure_ascii=False, default=str))

        # Guard: LLM 有时可能返回空字符串，为了避免消息校验失败，这里使用兜底文案
        is_fallback_response = False
        if not assistant_response or not assistant_response.strip():
            logger.warning(
                "Assistant response is empty; using fallback message to satisfy validation (session_id=%s)",
                session_id,
            )
            assistant_response = "当前没有找到与您问题相关的内容，请尝试换个问法或提供更多信息。"
            is_fallback_response = True

        # 构建 sources 并发送（按照文档格式，复用 chatbot.py 的逻辑）
        max_sources = int(os.getenv("CHATBOT_TOP_SOURCES", "5"))
        # 使用 build_chat_evidence 构建 evidence（与 chatbot.py 保持一致）
        graph_store = None
        try:
            graph_store = rag_inference_handler.get_graph_store()
        except Exception:  # noqa: BLE001
            graph_store = None
        evidence = build_chat_evidence(
            chunks or [],
            subgraph_data=subgraph_data,
            subgraph_info=subgraph_info,
            max_chunks=min(max_sources, CHAT_TOP_CHUNKS),
            graph_store=graph_store,
        )
        evidence_chunks = evidence.get("chunks") or []
        logger.info("SSE evidence_chunks count: %d", len(evidence_chunks))
        if evidence_chunks:
            first_content = str(evidence_chunks[0].get("content", ""))
            logger.info("SSE first evidence_chunk: id=%s, content_length=%d, content_preview=%s", 
                        evidence_chunks[0].get("id", "N/A"),
                        len(first_content),
                        first_content[:100] if first_content else '(empty)')
        
        # 使用 chatbot.py 中的函数构建 sources
        sources_for_frontend = await get_thread_pool().run_blocking(
            _build_sources_for_frontend,
            evidence_chunks,
            min(max_sources, CHAT_TOP_CHUNKS),
        )
        logger.info("SSE built %d sources for frontend", len(sources_for_frontend))
        
        # 保存 sources 到数据库（在过滤之前，保存完整的 sources 信息）
        sources_for_storage = [s.model_dump() for s in sources_for_frontend] if sources_for_frontend else None

        assistant_message = ChatMessage(
            session_id=session_id,
            content={"role": "assistant", "content": assistant_response},
            source_file_ids=[chunk.id for chunk in chunks] if chunks else None,
            sources=sources_for_storage,
            subgraph_data=subgraph_data if return_subgraph else None,
            raw_llm_response=raw_llm_response,
            raw_mindmap_response={"response": raw_mindmap_response} if raw_mindmap_response else None,
            created_at=datetime.now(),
        )
        assistant_message = await get_thread_pool().run_blocking(
            message_handler.create_message, assistant_message
        )

        # 如果是第一轮对话，生成并发送title（参考chatbot.py的实现）
        if first_turn:
            try:
                title = await _generate_title_via_llm(
                    rag_inference_handler,
                    query.strip(),
                    assistant_response.strip(),
                )
                # 发送title事件（使用统一的StandardResponse格式）
                yield sse_json_wrapped(
                    {"type": "title", "title": title},
                    request_id=request_id
                )
                # 更新数据库中的session name字段
                await get_thread_pool().run_blocking(
                    get_session_handler().update_session,
                    session_id,
                    {"name": title},
                )
                logger.info("SSE generated and updated session title: session_id=%s, title=%s", session_id, title)
            except Exception as exc:
                logger.warning("Failed to generate title: %s", exc)
                # title生成失败不影响主流程，继续执行
        if sources_for_frontend:
            for i, source in enumerate(sources_for_frontend):
                desc = getattr(source, 'description', '') or ''
                logger.info("SSE source[%d]: title=%s, description_length=%d, description_preview=%s", 
                            i,
                            getattr(source, 'title', 'N/A') or 'N/A',
                            len(desc),
                            desc[:100] if desc else '(empty)')
        
        # 根据回答中的 <sup> 标签过滤 sources（只返回被引用的）
        # 如果 LLM 响应为空（使用兜底消息），返回所有 sources，而不是空列表
        if is_fallback_response:
            logger.info("SSE using fallback response, returning all %d sources (no filtering)", len(sources_for_frontend))
        else:
            sources_for_frontend = _filter_sources_by_sup_keys(sources_for_frontend, assistant_response)
            logger.info("SSE filtered to %d sources after sup tag filtering", len(sources_for_frontend))
        
        # 发送 sources 事件（使用统一的 StandardResponse 格式）
        session_id_str = str(session_id)
        yield sse_json_wrapped(
            {
                "type": "sources",
                "sources": [s.model_dump() for s in sources_for_frontend],
                "id": session_id_str
            },
            request_id=request_id
        )
        logger.info("SSE sent sources event with %d sources", len(sources_for_frontend))

        # Align with WebSocket behavior: always build and return a payload.
        # When evidence/subgraph is requested, include the corresponding fields.
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

        # Build payload (aligned with WebSocket behavior).
        payload = _build_stream_chat_payload(
            assistant_message,
            chunks,
            subgraph=subgraph_data if return_subgraph else None,
            evidence=evidence,
        )

        # Return payload via tool_calls (OpenAI-compatible format).
        tool_calls = [
            {
                "index": 0,
                "id": f"call_{assistant_message.id}",
                "type": "function",
                "function": {
                    "name": "rag_arc_payload",
                    "arguments": json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")),
                },
            }
        ]
        yield sse_json_wrapped(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(role=None, tool_calls=tool_calls),
            ),
            request_id=request_id
        )

        yield sse_json_wrapped(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(),
                finish_reason="stop",
            ),
            request_id=request_id
        )
        yield sse_done()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@router.get("/stream_chat/{session_id}")
async def stream_chat_sse_get(
    session_id: uuid.UUID,
    query: str,
    current_user: Annotated[User | None, Depends(get_current_user)],
    return_subgraph: bool = False,
    target_owner_id: uuid.UUID | None = None,
    include_all_owners: bool = False,
    include_evidence: bool = False,
):
    """Backward compatible GET variant of the SSE stream chat endpoint."""

    request = StreamChatRequest(
        query=query,
        return_subgraph=return_subgraph,
        target_owner_id=target_owner_id,
        include_all_owners=include_all_owners,
        include_evidence=include_evidence,
    )
    return await stream_chat_sse(session_id=session_id, request=request, current_user=current_user)


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

            # Guard: only livingKB users (type=0) may request subgraph/evidence generation.
            if return_subgraph or include_evidence:
                user_type = getattr(current_user, "type", 0)
                if user_type != 0:
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION,
                        reason="Only livingKB users (type=0) can request subgraph generation"
                    )
                    return

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

            return_subgraph_flag = return_subgraph or include_evidence
            try:
                result = await rag_inference_handler.chat_async(
                    history_text,
                    owner_id=effective_owner,
                    return_subgraph=return_subgraph_flag,
                    current_user_query=query,
                )
            except TypeError:
                result = await rag_inference_handler.chat_async(
                    history_text,
                    owner_id=effective_owner,
                    return_subgraph=return_subgraph_flag,
                )

            raw_llm_response = None
            raw_mindmap_response = None
            if isinstance(result, tuple) and len(result) == 4:
                assistant_response, chunks, subgraph_data, subgraph_info = result
            else:
                (
                    assistant_response,
                    chunks,
                    subgraph_data,
                    subgraph_info,
                    raw_llm_response,
                    raw_mindmap_response,
                ) = result
            
            # Log full response details (including graph payload).
            logger.info(
                "WebSocket chat response: text_length=%d, chunks_count=%d, subgraph_nodes=%d, subgraph_edges=%d, raw_response=%s, raw_mindmap_response=%s",
                len(assistant_response) if assistant_response else 0,
                len(chunks) if chunks else 0,
                len(subgraph_data.get("nodes", [])) if subgraph_data else 0,
                len(subgraph_data.get("edges", [])) if subgraph_data else 0,
                json.dumps(raw_llm_response, ensure_ascii=False, default=str) if raw_llm_response else None,
                raw_mindmap_response if raw_mindmap_response else None
            )
            if subgraph_data:
                logger.debug("WebSocket subgraph data: %s", json.dumps(subgraph_data, ensure_ascii=False, default=str))
            
            assistant_message = ChatMessage(
                session_id=session_id,
                content={"role": "assistant", "content": assistant_response},
                source_file_ids=[chunk.id for chunk in chunks] if chunks else None,
                subgraph_data=subgraph_data if return_subgraph else None,
                raw_llm_response=raw_llm_response,
                raw_mindmap_response={"response": raw_mindmap_response} if raw_mindmap_response else None,
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
