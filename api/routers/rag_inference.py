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
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # Extract parameters from request body
    query = request.query
    return_subgraph = request.return_subgraph
    target_owner_id = request.target_owner_id
    include_all_owners = request.include_all_owners
    include_evidence = request.include_evidence
    enable_web_search = bool(getattr(request, "enable_web_search", False))
    enable_deepsearch = bool(getattr(request, "enable_deepsearch", False))

    # When DeepSearch is enabled, web search must also be enabled.
    if enable_deepsearch and not enable_web_search:
        enable_web_search = True
        logger.info("DeepSearch enabled; forcing web search on (enable_web_search=True)")
    else:
        logger.info("Web search enabled=%s", enable_web_search)

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
    if session is None:
        logger.warning("Session not found: session_id=%s, user_id=%s", session_id, current_user.id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=f"Session {session_id} not found or you don't have permission to access it"
        )
    if not validate_user_session(session, current_user):
        logger.warning("Session validation failed: session_id=%s, session_user_id=%s, current_user_id=%s", 
                      session_id, session.user_id, current_user.id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=f"Session {session_id} does not belong to current user. Session belongs to user {session.user_id}, but current user is {current_user.id}"
        )

    message_handler = get_message_handler()
    rag_inference_handler = get_rag_inference_handler()

    # Determine default owner scope based on user type (chatKB vs livingKB).
    effective_owner: uuid.UUID | None = get_default_owner_id(current_user)
    
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
        # Use correlation_id when available so request_id stays consistent with logs.
        request_id = correlation_id.get() or uuid.uuid4().hex
        progress_seq = 0

        # Qwen/OpenAI-compatible streams typically start with a chunk that sets role=assistant.
        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(role="assistant", content=""),
            ),
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
        
        # Determine whether this is the first turn (no assistant messages yet).
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

        # Queue for DeepSearch progress events (streamed in real time during DeepSearch execution).
        deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        
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
        
        # Helper for DeepSearch progress events (push to the dedicated queue for near real-time streaming).
        async def _emit_deepsearch_progress(payload: dict[str, Any]) -> None:
            """Emit DeepSearch progress events via the dedicated queue (bypasses the main queue)."""
            nonlocal progress_seq
            progress_seq += 1
            envelope = dict(payload or {})
            envelope.setdefault("v", 1)
            envelope.setdefault("type", "progress")
            envelope.setdefault("ts_ms", int(time.time() * 1000))
            envelope.setdefault("request_id", request_id)
            envelope.setdefault("seq", progress_seq)
            await deepsearch_progress_queue.put(envelope)

        # If DeepSearch is enabled, run it first to produce augmented context.
        enhanced_query = query
        if enable_deepsearch:
            logger.info("DeepSearch enabled for query: %s (owner_id=%s)", query, effective_owner)
            try:
                registrator = Register()
                deepsearch_service = registrator.get_object("deepsearch_service")
                logger.info("DeepSearch service found, running...")
                
                # Create a DeepSearch stage listener to stream progress via SSE.
                # Per DeepSearch docs, the stage order is:
                # created → planned → reasoned → gap_evaluated → external_invoked → reported → quality_gated → done/failed
                def deepsearch_stage_listener(record: dict[str, Any], state) -> None:
                    """DeepSearch stage-change listener that emits progress events via SSE."""
                    try:
                        stage = record.get("stage", "unknown")
                        metadata = record.get("metadata", {})
                        
                        logger.info("DeepSearch stage listener called: stage=%s, request_id=%s", stage, request_id)
                        
                        # Build progress payload (legacy SSE format: event=progress).
                        progress_info = {
                            "stage": "deepsearch",
                            "deepsearch_stage": stage,
                            "status": "running",
                        }
                        
                        # Match per documented stage names.
                        if stage == "planned":
                            progress_info["message"] = "正在生成搜索计划..."
                            # Read plan step count from metadata.
                            if "step_count" in metadata:
                                progress_info["plan_steps_count"] = metadata.get("step_count")
                            # Read full plan content from state.
                            if hasattr(state, "plan_steps") and state.plan_steps:
                                progress_info["plan_steps"] = state.plan_steps
                                logger.info("DeepSearch plan steps: %s", state.plan_steps)
                            if hasattr(state, "plan_metadata") and state.plan_metadata:
                                progress_info["plan_metadata"] = state.plan_metadata
                                logger.info("DeepSearch plan metadata: plan_id=%s, mode=%s, artifact_path=%s", 
                                          state.plan_metadata.get("plan_id"),
                                          state.plan_metadata.get("mode"),
                                          state.plan_metadata.get("artifact_path"))
                        elif stage == "reasoned":
                            progress_info["message"] = "正在进行图谱推理..."
                            # Read full reasoning trace from state.
                            if hasattr(state, "reasoning_trace") and state.reasoning_trace:
                                progress_info["reasoning_trace"] = state.reasoning_trace
                                
                                # Reasoning steps.
                                reasoning_steps = state.reasoning_trace.get("reasoning_steps", [])
                                if reasoning_steps:
                                    progress_info["reasoning_steps"] = reasoning_steps
                                    progress_info["reasoning_steps_count"] = len(reasoning_steps)
                                    logger.info("DeepSearch reasoning steps: count=%d, steps=%s", 
                                              len(reasoning_steps), reasoning_steps)
                                
                                # Tool call results.
                                tool_results = state.reasoning_trace.get("tool_results", [])
                                if tool_results:
                                    progress_info["tool_results"] = tool_results
                                    progress_info["tool_calls_count"] = len(tool_results)
                                    logger.info("DeepSearch tool results: count=%d, results=%s", 
                                              len(tool_results), tool_results)
                                    # Include the last tool call name if present.
                                    last_tool = tool_results[-1] if tool_results else {}
                                    if isinstance(last_tool, dict):
                                        tool_name = last_tool.get("tool_name", "")
                                        if tool_name:
                                            progress_info["last_tool"] = tool_name
                                
                                # Evidence.
                                evidences = state.reasoning_trace.get("evidences", [])
                                if evidences:
                                    progress_info["evidences"] = evidences
                                    progress_info["evidence_count"] = len(evidences)
                                    logger.info("DeepSearch evidences: count=%d", len(evidences))
                                
                                # Attach supplemental metadata.
                                if "evidence_count" in metadata:
                                    progress_info["evidence_count"] = metadata.get("evidence_count")
                                if "completed_steps" in metadata:
                                    progress_info["completed_steps"] = metadata.get("completed_steps")
                                    logger.info("DeepSearch completed steps: %s", metadata.get("completed_steps"))
                        elif stage == "gap_evaluated":
                            progress_info["message"] = "正在检测知识缺口..."
                            # Read full gap-evaluation result from state.
                            if hasattr(state, "gap_result") and state.gap_result:
                                progress_info["gap_result"] = state.gap_result
                                logger.info("DeepSearch gap result: %s", state.gap_result)
                            
                            # Attach gap-evaluation metadata.
                            if "should_trigger_external" in metadata:
                                progress_info["should_trigger_external"] = metadata.get("should_trigger_external")
                                logger.info("DeepSearch should_trigger_external: %s", metadata.get("should_trigger_external"))
                            if "reason" in metadata:
                                progress_info["gap_reason"] = metadata.get("reason")
                                logger.info("DeepSearch gap reason: %s", metadata.get("reason"))
                        elif stage == "external_invoked":
                            progress_info["message"] = "正在进行外部搜索..."
                            # Read external invocation info from state.
                            if hasattr(state, "external_calls") and state.external_calls:
                                progress_info["external_calls"] = state.external_calls
                                progress_info["external_calls_count"] = len(state.external_calls)
                                logger.info("DeepSearch external calls: count=%d, calls=%s", 
                                          len(state.external_calls), state.external_calls)
                            if "total_calls" in metadata:
                                progress_info["external_calls_count"] = metadata.get("total_calls")
                                logger.info("DeepSearch total external calls: %s", metadata.get("total_calls"))
                        elif stage == "reported":
                            progress_info["message"] = "正在生成报告..."
                            # Read report payload from state.
                            if hasattr(state, "report_payload") and state.report_payload:
                                progress_info["report_payload"] = state.report_payload
                                
                                # Answer.
                                answer = state.report_payload.get("answer", "")
                                if answer:
                                    progress_info["answer"] = answer
                                    progress_info["answer_length"] = len(answer)
                                    logger.info("DeepSearch report answer: length=%d, preview=%s", 
                                              len(answer), answer[:200] if len(answer) > 200 else answer)
                                
                                # Structured report.
                                structured_report = state.report_payload.get("structured_report")
                                if structured_report:
                                    progress_info["structured_report"] = structured_report
                                    logger.info("DeepSearch structured report: %s", structured_report)
                                
                                # References and evidence.
                                references = state.report_payload.get("references", [])
                                if references:
                                    progress_info["references"] = references
                                    progress_info["references_count"] = len(references)
                                    logger.info("DeepSearch references: count=%d", len(references))
                            
                            # Attach report metadata.
                            if "answer_length" in metadata:
                                progress_info["answer_length"] = metadata.get("answer_length")
                            if "evidence_count" in metadata:
                                progress_info["evidence_count"] = metadata.get("evidence_count")
                                logger.info("DeepSearch evidence count: %s", metadata.get("evidence_count"))
                        elif stage == "quality_gated":
                            progress_info["message"] = "正在进行质量检查..."
                            # Read quality-gate info from state.
                            if hasattr(state, "quality_gates") and state.quality_gates:
                                progress_info["quality_gates"] = state.quality_gates
                                progress_info["quality_gates_count"] = len(state.quality_gates)
                                logger.info("DeepSearch quality gates: count=%d, gates=%s", 
                                          len(state.quality_gates), state.quality_gates)
                            
                            # Attach quality-gate metadata.
                            if "passed" in metadata:
                                progress_info["quality_passed"] = metadata.get("passed")
                                logger.info("DeepSearch quality passed: %s", metadata.get("passed"))
                            if "should_iterate" in metadata:
                                progress_info["should_iterate"] = metadata.get("should_iterate")
                                logger.info("DeepSearch should iterate: %s", metadata.get("should_iterate"))
                            if "round" in metadata:
                                progress_info["round"] = metadata.get("round")
                                logger.info("DeepSearch quality gate round: %s", metadata.get("round"))
                        elif stage == "done":
                            progress_info["status"] = "completed"
                            progress_info["message"] = "DeepSearch 完成"
                            # Attach completion summary.
                            if hasattr(state, "run_id"):
                                progress_info["run_id"] = state.run_id
                                logger.info("DeepSearch completed: run_id=%s", state.run_id)
                            if hasattr(state, "cost_telemetry") and state.cost_telemetry:
                                progress_info["cost_telemetry"] = state.cost_telemetry
                                logger.info("DeepSearch cost telemetry: %s", state.cost_telemetry)
                            if hasattr(state, "stage_history") and state.stage_history:
                                progress_info["stage_history"] = state.stage_history
                                logger.info("DeepSearch stage history: count=%d", len(state.stage_history))
                        elif stage == "failed":
                            progress_info["status"] = "failed"
                            progress_info["message"] = "DeepSearch 执行失败"
                            # Attach failure details.
                            if hasattr(state, "errors") and state.errors:
                                progress_info["errors"] = state.errors
                                logger.error("DeepSearch errors: %s", state.errors)
                            if hasattr(state, "run_id"):
                                progress_info["run_id"] = state.run_id
                                logger.error("DeepSearch failed: run_id=%s", state.run_id)
                        elif stage == "created":
                            progress_info["message"] = "DeepSearch 初始化..."
                            # Attach initialization details.
                            if hasattr(state, "run_id"):
                                progress_info["run_id"] = state.run_id
                                logger.info("DeepSearch initialized: run_id=%s", state.run_id)
                            if hasattr(state, "config_fingerprint"):
                                progress_info["config_fingerprint"] = state.config_fingerprint
                                logger.info("DeepSearch config fingerprint: %s", state.config_fingerprint)
                        else:
                            # Unknown stage; still record it.
                            progress_info["message"] = f"DeepSearch 执行中（阶段: {stage}）..."
                        
                        # Attach common metadata.
                        if metadata:
                            # Plan metadata.
                            if "plan_id" in metadata:
                                progress_info["plan_id"] = metadata.get("plan_id")
                            
                        # Emit via dedicated queue (bypasses the main queue).
                        asyncio.run_coroutine_threadsafe(_emit_deepsearch_progress(progress_info), loop)
                        logger.info("DeepSearch progress emitted: stage=%s, message=%s", stage, progress_info.get("message"))
                    except Exception as e:
                        logger.error("DeepSearch stage listener error: %s", e, exc_info=True)
                
                logger.info("Passing stage_listener to DeepSearch service (request_id=%s)", request_id)
                
                # Create a task to run DeepSearch in parallel.
                deepsearch_task = asyncio.create_task(
                    deepsearch_service.run(
                        question=query,
                        owner_id=str(effective_owner),
                        stage_listener=deepsearch_stage_listener,
                    )
                )
                
                # While DeepSearch is running, consume and emit progress events in real time.
                # Use a dedicated queue so progress is not delayed by the main queue.
                deepsearch_progress_task = None
                while not deepsearch_task.done():
                    # Create the queue wait task if missing or already completed.
                    if deepsearch_progress_task is None or deepsearch_progress_task.done():
                        deepsearch_progress_task = asyncio.create_task(deepsearch_progress_queue.get())
                    
                    # Wait on both: DeepSearch completion and the next progress event.
                    done, pending = await asyncio.wait(
                        [deepsearch_task, deepsearch_progress_task],
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Process progress events if available.
                    if deepsearch_progress_task in done:
                        try:
                            progress_payload = await deepsearch_progress_task
                            deepsearch_progress_task = None  # Reset; recreate on next iteration.
                            
                            if progress_payload is None:
                                # Received sentinel; keep waiting for DeepSearch completion.
                                continue
                            
                            # Yield the DeepSearch progress event directly.
                            logger.info("Yielding DeepSearch progress SSE: stage=%s, request_id=%s", progress_payload.get("deepsearch_stage"), request_id)
                            tool_calls = [
                                {
                                    "index": 0,
                                    "id": f"call_deepsearch_progress_{uuid.uuid4().hex}",
                                    "type": "function",
                                    "function": {
                                        "name": "rag_arc_progress",
                                        "arguments": json.dumps(
                                            progress_payload,
                                            ensure_ascii=False,
                                            default=str,
                                            separators=(",", ":"),
                                        ),
                                    },
                                }
                            ]
                            yield sse_json(
                                openai_chat_completion_chunk(
                                    chunk_id=chunk_id,
                                    model=model_name,
                                    created=created,
                                    delta=delta_envelope(role=None, tool_calls=tool_calls),
                                ),
                            )
                        except Exception as e:
                            logger.error("Error processing DeepSearch progress: %s", e, exc_info=True)
                            deepsearch_progress_task = None
                    
                    # If DeepSearch finished, exit the loop.
                    if deepsearch_task in done:
                        # Cancel the progress wait task (if still waiting).
                        if deepsearch_progress_task and not deepsearch_progress_task.done():
                            deepsearch_progress_task.cancel()
                            try:
                                await deepsearch_progress_task
                            except asyncio.CancelledError:
                                pass
                        break
                
                # Get DeepSearch result.
                deepsearch_result = await deepsearch_task
                logger.info("DeepSearch service returned (request_id=%s)", request_id)
                
                # After DeepSearch completes, drain remaining progress events (up to 1 second).
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    try:
                        progress_payload = await asyncio.wait_for(deepsearch_progress_queue.get(), timeout=0.1)
                        if progress_payload is None:
                            break
                        logger.info("Yielding remaining DeepSearch progress SSE: stage=%s, request_id=%s", progress_payload.get("deepsearch_stage"), request_id)
                        tool_calls = [
                            {
                                "index": 0,
                                "id": f"call_deepsearch_progress_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {
                                    "name": "rag_arc_progress",
                                    "arguments": json.dumps(
                                        progress_payload,
                                        ensure_ascii=False,
                                        default=str,
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ]
                        yield sse_json(
                            openai_chat_completion_chunk(
                                chunk_id=chunk_id,
                                model=model_name,
                                created=created,
                                delta=delta_envelope(role=None, tool_calls=tool_calls),
                            ),
                        )
                    except (asyncio.TimeoutError, asyncio.QueueEmpty):
                        break
                    except Exception as e:
                        logger.error("Error consuming remaining DeepSearch progress: %s", e, exc_info=True)
                        break
                
                deepsearch_answer = deepsearch_result.get("answer") or deepsearch_result.get("report") or ""
                if deepsearch_answer:
                    enhanced_query = f"{query}\n\n[DeepSearch 增强上下文]\n{deepsearch_answer}"
                    logger.info("DeepSearch completed successfully, enhanced query length: %d", len(enhanced_query))
                else:
                    logger.warning("DeepSearch completed but no answer/report found in result")
            except KeyError as e:
                logger.warning("DeepSearch service not available: %s", e)
            except Exception as e:
                logger.error("DeepSearch failed: %s", e, exc_info=True)

        def _run_stream() -> None:
            try:
                _emit_progress({"stage": "prepare", "status": "start"})
                
                # Read USER_TYPE from environment to select the corresponding prompt.
                user_type_str = os.getenv("USER_TYPE", "0")
                try:
                    user_type = int(user_type_str)
                except ValueError:
                    logger.warning(
                        "Invalid USER_TYPE environment variable: %s, defaulting to 0",
                        user_type_str,
                    )
                    user_type = 0

                try:
                    token_stream, chunks, subgraph_data, subgraph_info = rag_inference_handler.stream_chat(
                        enhanced_query,
                        effective_owner,
                        return_subgraph=(return_subgraph or include_evidence),
                        progress_callback=_emit_progress,
                        history_text=history_text if history_text else None,
                        enable_web_search=enable_web_search,
                        user_type=user_type,
                    )
                except TypeError:
                    # Backward compatibility: older implementations may not accept `history_text` / `enable_web_search` / `user_type`.
                    logger.info(
                        "stream_chat signature mismatch; falling back to reduced args",
                        exc_info=True,
                    )
                    try:
                        token_stream, chunks, subgraph_data, subgraph_info = rag_inference_handler.stream_chat(
                            enhanced_query,
                            effective_owner,
                            return_subgraph=(return_subgraph or include_evidence),
                            progress_callback=_emit_progress,
                            history_text=history_text if history_text else None,
                            enable_web_search=enable_web_search,
                        )
                    except TypeError:
                        try:
                            token_stream, chunks, subgraph_data, subgraph_info = rag_inference_handler.stream_chat(
                                enhanced_query,
                                effective_owner,
                                return_subgraph=(return_subgraph or include_evidence),
                                progress_callback=_emit_progress,
                                history_text=history_text if history_text else None,
                                user_type=user_type,
                            )
                        except TypeError:
                            token_stream, chunks, subgraph_data, subgraph_info = rag_inference_handler.stream_chat(
                                enhanced_query,
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
                yield sse_json(
                    openai_chat_completion_chunk(
                        chunk_id=chunk_id,
                        model=model_name,
                        created=created,
                        delta=delta_envelope(role=None, tool_calls=tool_calls),
                    ),
                )
                continue

            if isinstance(item, dict) and item.get("kind") == "token":
                piece = str(item.get("text") or "")
                if not piece:
                    continue
                for delta_piece in iter_text_deltas(piece):
                    response_parts.append(delta_piece)
                    yield sse_json(
                        openai_chat_completion_chunk(
                            chunk_id=chunk_id,
                            model=model_name,
                            created=created,
                            delta=delta_envelope(role=None, content=delta_piece),
                        ),
                    )
                    await asyncio.sleep(0)
                continue

        if stream_error[0] is not None:
            yield sse_json({"error": {"message": str(stream_error[0])}})
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

        # Guard: the LLM may return an empty response; use a fallback to satisfy validation.
        is_fallback_response = False
        if not assistant_response or not assistant_response.strip():
            logger.warning(
                "Assistant response is empty; using fallback message to satisfy validation (session_id=%s)",
                session_id,
            )
            assistant_response = "当前没有找到与您问题相关的内容，请尝试换个问法或提供更多信息。"
            is_fallback_response = True

        # Build sources and emit them (aligned with the chatbot.py behavior).
        max_sources = int(os.getenv("CHATBOT_TOP_SOURCES", "5"))
        # Build evidence via build_chat_evidence (keep behavior aligned with chatbot.py).
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
        
        # Reuse chatbot.py helper to build sources.
        sources_for_frontend = await get_thread_pool().run_blocking(
            _build_sources_for_frontend,
            evidence_chunks,
            min(max_sources, CHAT_TOP_CHUNKS),
        )
        logger.info("SSE built %d sources for frontend", len(sources_for_frontend))

        citation_key_map: dict[int, int] = {}
        if is_fallback_response:
            logger.info(
                "SSE using fallback response; skip citation-driven source filtering/renumbering (sources=%d)",
                len(sources_for_frontend),
            )
        else:
            assistant_response, sources_for_frontend, citation_key_map = _filter_and_renumber_sources_by_sup_keys_sorted(
                sources_for_frontend,
                assistant_response,
            )
            logger.info(
                "SSE citation normalization applied: cited_sources=%d citation_key_map=%s",
                len(sources_for_frontend),
                citation_key_map,
            )

        sources_for_storage = [s.model_dump() for s in sources_for_frontend] if sources_for_frontend else None

        assistant_message = ChatMessage(
            session_id=session_id,
            content={"role": "assistant", "content": assistant_response},
            source_file_ids=[s.chunk_id for s in sources_for_frontend] if sources_for_frontend else None,
            sources=sources_for_storage,
            subgraph_data=subgraph_data if return_subgraph else None,
            raw_llm_response=raw_llm_response,
            raw_mindmap_response={"response": raw_mindmap_response} if raw_mindmap_response else None,
            created_at=datetime.now(),
        )
        assistant_message = await get_thread_pool().run_blocking(
            message_handler.create_message, assistant_message
        )

        # If this is the first turn, generate and emit a title (aligned with chatbot.py).
        if first_turn:
            try:
                title = await generate_title_via_llm(
                    rag_inference_handler,
                    query.strip(),
                    assistant_response.strip(),
                )
                yield sse_json({"type": "title", "title": title})
                # Update session name in DB.
                await get_thread_pool().run_blocking(
                    get_session_handler().update_session,
                    session_id,
                    {"name": title},
                )
                logger.info("SSE generated and updated session title: session_id=%s, title=%s", session_id, title)
            except Exception as exc:
                logger.warning("Failed to generate title: %s", exc)
                # Title generation failure should not break the main flow.
        if sources_for_frontend:
            for i, source in enumerate(sources_for_frontend):
                desc = getattr(source, 'description', '') or ''
                logger.info("SSE source[%d]: title=%s, description_length=%d, description_preview=%s", 
                            i,
                            getattr(source, 'title', 'N/A') or 'N/A',
                            len(desc),
                            desc[:100] if desc else '(empty)')
        
        # Emit the sources event (plain JSON).
        session_id_str = str(session_id)
        yield sse_json(
            {
                "type": "sources",
                "sources": [s.model_dump() for s in sources_for_frontend],
                "citation_key_map": {str(k): v for k, v in (citation_key_map or {}).items()},
                "id": session_id_str
            },
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
        payload = build_stream_chat_payload(
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
        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(role=None, tool_calls=tool_calls),
            ),
        )

        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(),
                finish_reason="stop",
            ),
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

            # Always enable web search for WebSocket stream_chat.
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

            response_payload = build_stream_chat_payload(
                assistant_message,
                chunks,
                subgraph=subgraph_data if return_subgraph else None,
                evidence=evidence,
            )
            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnect (session_id=%s user=%s)", session_id, current_user.id)
