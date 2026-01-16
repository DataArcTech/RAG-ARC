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
from asgi_correlation_id import correlation_id
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
    _sse_json,
    _sse_done,
    _build_sources_for_frontend,
    _filter_and_renumber_sources_by_sup_keys_sorted,
)
from encapsulation.data_model.orm_models import ChatMessage, User
from encapsulation.data_model.schema import Chunk, GraphData
from framework.thread_pool import get_thread_pool
from framework.register import Register
import uuid
import logging
from core.utils.owner_guard import is_admin_owner, get_admin_owner_id
from core.presentation.evidence import build_chat_evidence
from config.output_limits import CHAT_TOP_CHUNKS
from api.routers.rag_inference_handlers import (
    generate_title_via_llm,
    get_account_handler,
    get_default_owner_id,
    get_message_handler,
    get_rag_inference_handler,
    get_session_handler,
)
from api.routers.rag_inference_models import (
    ChatRequest,
    ChatResponse,
    GraphOverviewResponse,
    StreamChatRequest,
    build_stream_chat_payload,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


router = APIRouter(prefix="/rag_inference", tags=["rag_inference"])

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

    # Guard: only livingKB users (type=0) may request subgraph/evidence generation.
    if request.return_subgraph or request.include_evidence:
        user_type = getattr(current_user, "type", 0)
        if user_type != 0:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only livingKB users (type=0) can request subgraph generation"
            )

    # Determine default owner scope based on user type (chatKB vs livingKB).
    effective_owner_id: uuid.UUID | None = get_default_owner_id(current_user)

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
    try:
        response_text, chunks, subgraph_data, subgraph_info, raw_llm_response, raw_mindmap_response = await rag_inference_handler.chat_async(
            request.query,
            owner_id=effective_owner_id,
            return_subgraph=needs_subgraph,
            current_user_query=request.query,
            enable_web_search=bool(getattr(request, "enable_web_search", False)),
        )
    except TypeError:
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
    from api.routers.rag_inference_modules.stream_chat import (
        validators,
        owner_resolver,
        event_generator as stream_chat_event_generator,
    )
    
    # Validate authentication and permissions
    await validators.validate_user_authentication(current_user)
    await validators.validate_user_permissions(
        current_user,
        request.return_subgraph,
        request.include_evidence
    )
    await validators.validate_session_access(session_id, current_user)
    
    # Extract parameters
    query = request.query
    return_subgraph = request.return_subgraph
    target_owner_id = request.target_owner_id
    include_all_owners = request.include_all_owners
    include_evidence = request.include_evidence
    enable_deepsearch = bool(getattr(request, "enable_deepsearch", False))
    enable_web_search = bool(getattr(request, "enable_web_search", False))
    
    # DeepSearch 和 Web Search 现在互相独立
    if enable_deepsearch:
        logger.info("DeepSearch enabled (enable_deepsearch=True)")
    if enable_web_search:
        logger.info("Web search enabled (enable_web_search=True)")
    
    logger.info("SSE stream_chat request for session_id %s by user %s", session_id, current_user.id)
    
    # 检查是否有正在进行的任务
    from api.routers.rag_inference_modules.stream_chat.task.task_registry import get_chat_task_registry
    registry = get_chat_task_registry()
    existing_task = await registry.get_by_session(session_id)
    
    if existing_task and not existing_task.done and not existing_task.cancelled:
        # 有未完成的任务，恢复任务（不取消，因为可能是同一个请求的重连）
        logger.info("Resuming task: task_id=%s, session_id=%s", existing_task.task_id, session_id)
        return await _resume_task_sse(existing_task, request_id=correlation_id.get() or uuid.uuid4().hex)
    
    # Resolve effective owner
    effective_owner = owner_resolver.resolve_effective_owner(
        current_user,
        target_owner_id,
        include_all_owners
    )
    
    model_name = os.getenv("CHAT_MODEL_NAME") or os.getenv("OPENAI_CHAT_MODEL") or "rag-arc"
    request_id = correlation_id.get() or uuid.uuid4().hex

    async def event_generator():
        async for event in stream_chat_event_generator.generate_sse_events(
            session_id=session_id,
            query=query,
            effective_owner=effective_owner,
            return_subgraph=return_subgraph,
            include_evidence=include_evidence,
            enable_web_search=enable_web_search,
            enable_deepsearch=enable_deepsearch,
            model_name=model_name,
            request_id=request_id,
        ):
            yield event

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


async def _resume_task_sse(
    task_info,
    request_id: str
):
    """Resume a task by replaying cached events and continuing stream."""
    from fastapi.responses import StreamingResponse
    from api.routers.rag_inference_modules.stream_chat.task.task_registry import get_chat_task_registry
    from api.routers.rag_inference_modules.stream_chat.task.task_resume import (
        _yield_task_resumed_event,
        _yield_completed_task_response,
        _replay_cached_events,
        _poll_task_updates,
        _yield_task_completion_event
    )
    
    async def event_generator():
        try:
            logger.info("Starting event_generator for task resume: task_id=%s, request_id=%s", task_info.task_id, request_id)
            # 重新获取最新任务状态（因为任务可能在恢复过程中完成）
            registry = get_chat_task_registry()
            updated_task_info = await registry.get(task_info.task_id)
            if not updated_task_info:
                logger.warning("Task %s not found during resume", task_info.task_id)
                yield sse_json_wrapped({"error": {"message": "Task not found"}}, request_id=request_id, code=404, message="error")
                yield sse_done()
                return
            
            current_task_info = updated_task_info
            logger.info("Task %s found: done=%s, cancelled=%s, response_length=%d, assistant_message_id=%s", 
                       current_task_info.task_id, current_task_info.done, current_task_info.cancelled, 
                       current_task_info.response_length, current_task_info.assistant_message_id)
            
            # 1. 发送任务恢复通知
            logger.info("Yielding task_resumed event for task %s", current_task_info.task_id)
            async for event in _yield_task_resumed_event(current_task_info, request_id):
                yield event
            logger.info("Task_resumed event sent for task %s", current_task_info.task_id)
        
            # 2. 如果任务已完成，直接返回最终结果
            # 检查任务是否已完成：done 标志或 assistant_message_id 存在
            is_completed = current_task_info.done or current_task_info.assistant_message_id is not None
            if is_completed:
                if not current_task_info.done and current_task_info.assistant_message_id:
                    logger.info("Task %s appears completed (has assistant_message_id) but done flag not set, marking as done", 
                               current_task_info.task_id)
                    # 确保任务标记为完成
                    await registry.mark_done(current_task_info.task_id, assistant_message_id=current_task_info.assistant_message_id)
                    # 重新获取更新后的任务信息
                    current_task_info = await registry.get(current_task_info.task_id) or current_task_info
                
                logger.info("Task %s is done, sending completed response", current_task_info.task_id)
                async for event in _yield_completed_task_response(current_task_info, request_id):
                    yield event
                logger.info("Completed response sent for task %s", current_task_info.task_id)
                return
            
            # 3. 回放已缓存的事件
            logger.info("Replaying cached events for task %s", current_task_info.task_id)
            try:
                async for event in _replay_cached_events(current_task_info, request_id):
                    yield event
                logger.info("Cached events replayed for task %s", current_task_info.task_id)
            except Exception as e:
                logger.warning("Error during replay cached events for task %s: %s", current_task_info.task_id, e, exc_info=True)
                # 即使出错，也继续执行后续逻辑
            
            # 4. 再次检查任务状态（可能在回放期间完成）
            updated_task_info = await registry.get(current_task_info.task_id)
            if updated_task_info:
                # 检查任务是否已完成：done 标志或 assistant_message_id 存在
                is_completed = updated_task_info.done or updated_task_info.assistant_message_id is not None
                
                # 如果任务有响应内容但没有assistant_message_id，检查数据库中是否有对应的assistant message
                if not is_completed and updated_task_info.response_length > 0 and updated_task_info.user_message_id:
                    try:
                        from api.routers.rag_inference_handlers import get_message_handler
                        from framework.thread_pool import get_thread_pool
                        
                        # 检查数据库中是否有对应的assistant message（在user_message之后创建的）
                        message_handler = get_message_handler()
                        messages = await get_thread_pool().run_blocking(
                            message_handler.list_messages_by_session,
                            updated_task_info.session_id,
                            limit=50  # 检查最近50条消息
                        )
                        
                        # 查找user_message之后的assistant message
                        # 消息按 created_at 升序排列（oldest first），所以 assistant message 在 user_message 之后
                        user_message_index = None
                        for i, msg in enumerate(messages):
                            if str(msg.id) == str(updated_task_info.user_message_id):
                                user_message_index = i
                                break
                        
                        # 如果找到user_message，检查它之后的assistant message
                        if user_message_index is not None:
                            # 在user_message之后（索引更大）查找assistant message
                            for i in range(user_message_index + 1, len(messages)):
                                msg = messages[i]
                                if msg.content.get("role") == "assistant":
                                    logger.info("Task %s has response but no assistant_message_id, found assistant message %s in database, marking as done", 
                                               updated_task_info.task_id, msg.id)
                                    await registry.mark_done(updated_task_info.task_id, assistant_message_id=msg.id)
                                    updated_task_info = await registry.get(updated_task_info.task_id) or updated_task_info
                                    is_completed = True
                                    break
                    except Exception as e:
                        logger.warning("Failed to check database for assistant message for task %s: %s", 
                                     updated_task_info.task_id, e)
                
                if is_completed:
                    if not updated_task_info.done and updated_task_info.assistant_message_id:
                        logger.info("Task %s appears completed (has assistant_message_id) but done flag not set, marking as done", 
                                   updated_task_info.task_id)
                        await registry.mark_done(updated_task_info.task_id, assistant_message_id=updated_task_info.assistant_message_id)
                        updated_task_info = await registry.get(updated_task_info.task_id) or updated_task_info
                    
                    logger.info("Task %s completed during replay, sending completion event", updated_task_info.task_id)
                    async for event in _yield_task_completion_event(updated_task_info, request_id):
                        yield event
                    logger.info("Completion event sent for task %s", updated_task_info.task_id)
                    return
                current_task_info = updated_task_info
            
            # 5. 任务还在进行中，继续监听新事件（轮询）
            logger.info("Task %s still processing, polling for updates", current_task_info.task_id)
            async for event in _poll_task_updates(current_task_info, request_id):
                yield event
            logger.info("Polling completed for task %s", current_task_info.task_id)
            
            # 6. 任务完成
            final_task_info = await registry.get(current_task_info.task_id)
            if final_task_info:
                logger.info("Sending final completion event for task %s", final_task_info.task_id)
                async for event in _yield_task_completion_event(final_task_info, request_id):
                    yield event
                logger.info("Final completion event sent for task %s", final_task_info.task_id)
        except Exception as e:
            logger.error("Error in event_generator for task resume: task_id=%s, error=%s", task_info.task_id, str(e), exc_info=True)
            yield sse_json_wrapped({"error": {"message": f"Resume failed: {str(e)}"}}, request_id=request_id, code=500, message="error")
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
    enable_web_search: bool = False,
):
    """Backward compatible GET variant of the SSE stream chat endpoint."""

    request = StreamChatRequest(
        query=query,
        return_subgraph=return_subgraph,
        target_owner_id=target_owner_id,
        include_all_owners=include_all_owners,
        include_evidence=include_evidence,
        enable_web_search=enable_web_search,
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
            enable_web_search = False
            query = message_text

            try:
                payload = json.loads(message_text)
                if isinstance(payload, dict):
                    query = payload.get("query") or payload.get("message") or query
                    return_subgraph = bool(payload.get("return_subgraph", False))
                    include_all_owners = bool(payload.get("include_all_owners", False))
                    include_evidence = bool(payload.get("include_evidence", False))
                    enable_web_search = bool(payload.get("enable_web_search", False))
                    if payload.get("target_owner_id"):
                        target_owner_id = uuid.UUID(str(payload["target_owner_id"]))
            except Exception:  # noqa: BLE001
                pass

            # 无论 DeepSearch 是否开启，都开启联网搜索
            enable_web_search = True
            logger.info("Web search enabled (enable_web_search=True)")

            # Guard: only livingKB users (type=0) may request subgraph/evidence generation.
            if return_subgraph or include_evidence:
                user_type = getattr(current_user, "type", 0)
                if user_type != 0:
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION,
                        reason="Only livingKB users (type=0) can request subgraph generation"
                    )
                    return

            effective_owner: uuid.UUID | None = get_default_owner_id(current_user)
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
                    enable_web_search=enable_web_search,
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

            response_payload, subgraph_for_outer = build_stream_chat_payload(
                assistant_message,
                chunks,
                subgraph=subgraph_data if return_subgraph else None,
                evidence=evidence,
            )
            # Add subgraph at outer level if present
            if subgraph_for_outer is not None:
                response_payload["subgraph"] = subgraph_for_outer
            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnect (session_id=%s user=%s)", session_id, current_user.id)


@router.post("/stream_chat/{session_id}/cancel")
async def cancel_chat_task(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Cancel the current chat task for a session."""
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    # 验证 session 权限
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    
    # 获取当前任务
    from api.routers.rag_inference_modules.stream_chat.task.task_registry import get_chat_task_registry
    from encapsulation.data_model.orm_models import ChatMessageStatus
    
    registry = get_chat_task_registry()
    task_info = await registry.get_by_session(session_id)
    
    if not task_info:
        return {"success": False, "message": "No active task found"}
    
    if task_info.done:
        return {"success": False, "message": "Task already completed"}
    
    # 取消任务
    cancelled = await registry.cancel(task_info.task_id, "User cancelled")
    
    if cancelled:
        # 更新数据库状态
        await get_thread_pool().run_blocking(
            get_message_handler().update_message,
            task_info.user_message_id,
            {"status": ChatMessageStatus.CANCELLED}
        )
        
        await get_thread_pool().run_blocking(
            get_session_handler().update_session,
            session_id,
            {
                "current_task_id": None,
                "current_task_status": None,
                "current_task_started_at": None
            }
        )
        
        return {"success": True, "message": "Task cancelled"}
    else:
        return {"success": False, "message": "Failed to cancel task"}


@router.get("/stream_chat/{session_id}/status")
async def get_chat_task_status(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Get the current task status for a session."""
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    from api.routers.rag_inference_modules.stream_chat.task.task_registry import get_chat_task_registry
    
    registry = get_chat_task_registry()
    
    # 快速获取当前session的任务（只返回最新的）
    task_info = await registry.get_by_session(session_id)
    
    if not task_info:
        return {
            "has_active_task": False,
            "status": None
        }
    
    return {
        "has_active_task": not task_info.done and not task_info.cancelled,
        "status": "cancelled" if task_info.cancelled else ("done" if task_info.done else "processing"),
        "task_id": task_info.task_id,
        "query": task_info.query,
        "created_at": task_info.created_at_ms,
        "events_count": task_info.events_count,
        "response_length": task_info.response_length
    }
