"""最终响应构建函数"""
import json
import logging
from typing import Any, Optional, AsyncGenerator, Tuple
from .response_builder import (
    generate_mindmap_if_needed,
    ensure_non_empty_response,
    build_sources_and_evidence,
    create_assistant_message,
    generate_and_update_title,
    build_evidence_for_payload,
)
from api.routers.rag_inference_models import build_stream_chat_payload
from api.sse import (
    openai_chat_completion_chunk,
    delta_envelope,
    sse_json_wrapped,
)
from ..task.task_helpers import check_and_handle_cancellation, yield_cancellation_event, mark_task_completed
from ..task.task_registry import get_chat_task_registry
from api.routers.rag_inference_handlers import get_message_handler
from framework.thread_pool import get_thread_pool

logger = logging.getLogger(__name__)

async def _get_or_create_final_assistant_message(
    *,
    session_id: Any,
    assistant_response: str,
    sources_for_frontend: list,
    subgraph_data: Optional[Any],
    return_subgraph: bool,
    raw_llm_response: Optional[Any],
    raw_mindmap_response: Optional[Any],
    task_info: Optional[Any],
    user_message: Any,
    deepsearch_trace_file_path: Optional[str],
) -> Tuple[Any, bool]:
    """Create the assistant message and return (message, created_now).
    
    注意：即使 return_subgraph=True，也可能因为 mindmap 生成失败/被取消等原因导致 subgraph_data 为 None。
    这时仍然允许创建消息（subgraph_data 留空），并在后续拿到 subgraph_data 时更新已有消息。
    """
    if return_subgraph and subgraph_data is None:
        logger.warning(
            "[STORE] return_subgraph=True but subgraph_data is None, checking for existing message"
        )
        if task_info:
            registry = get_chat_task_registry()
            lock = await registry.get_finalization_lock(task_info.task_id)
            async with lock:
                latest = await registry.get(task_info.task_id) or task_info
                existing_id = getattr(latest, "assistant_message_id", None)
                if existing_id is not None:
                    existing = await get_thread_pool().run_blocking(get_message_handler().get_message, existing_id)
                    if existing is not None:
                        logger.info(
                            "[STORE] Returning existing message (will update subgraph_data later): message_id=%s",
                            existing.id
                        )
                        return existing, False
        logger.warning(
            "[STORE] Creating message without subgraph_data (mindmap may have failed): session_id=%s, task_id=%s",
            session_id,
            task_info.task_id if task_info else "None",
        )
    
    # 直接创建新消息（此时 subgraph_data 已存在或 return_subgraph=False）
    if not task_info:
        msg = await create_assistant_message(
            session_id,
            assistant_response,
            sources_for_frontend,
            subgraph_data,
            return_subgraph,
            raw_llm_response,
            raw_mindmap_response,
            deepsearch_trace_file_path=deepsearch_trace_file_path,
        )
        return msg, True

    # 如果有 task_info，使用锁防止并发创建重复消息
    registry = get_chat_task_registry()
    lock = await registry.get_finalization_lock(task_info.task_id)
    async with lock:
        latest = await registry.get(task_info.task_id) or task_info
        
        # 检查是否已经存在消息（防止重复创建）
        existing_id = getattr(latest, "assistant_message_id", None)
        if existing_id is not None:
            existing = await get_thread_pool().run_blocking(get_message_handler().get_message, existing_id)
            if existing is not None:
                # 如果旧消息缺少 subgraph_data 或 sources，更新它
                needs_update = False
                updates = {}
                
                # 检查并准备更新 subgraph_data（优先更新，确保存储完整数据）
                if subgraph_data is not None and (existing.subgraph_data is None or existing.subgraph_data == {}):
                    updates["subgraph_data"] = subgraph_data
                    needs_update = True
                    logger.info(
                        "[STORE] Updating existing message subgraph_data: message_id=%s",
                        existing.id
                    )
                
                # 检查并准备更新 sources
                if sources_for_frontend and (existing.sources is None or len(existing.sources) == 0):
                    sources_for_storage = (
                        [s.model_dump() if hasattr(s, 'model_dump') else s for s in sources_for_frontend]
                        if sources_for_frontend
                        else None
                    )
                    if sources_for_storage:
                        updates["sources"] = sources_for_storage
                        needs_update = True
                        logger.info(
                            "[STORE] Updating existing message sources: message_id=%s, sources_count=%d",
                            existing.id,
                            len(sources_for_storage)
                        )
                
                # 检查并准备更新 source_file_ids
                if sources_for_frontend:
                    source_file_ids = [s.chunk_id for s in sources_for_frontend if hasattr(s, 'chunk_id')]
                    if source_file_ids and (existing.source_file_ids is None or len(existing.source_file_ids) == 0):
                        updates["source_file_ids"] = source_file_ids
                        needs_update = True
                        logger.info(
                            "[STORE] Updating existing message source_file_ids: message_id=%s, source_file_ids_count=%d",
                            existing.id,
                            len(source_file_ids)
                        )
                
                # 如果有需要更新的字段，执行更新
                if needs_update:
                    await get_thread_pool().run_blocking(
                        get_message_handler().update_message,
                        existing.id,
                        updates
                    )
                    # 重新获取更新后的消息
                    existing = await get_thread_pool().run_blocking(get_message_handler().get_message, existing_id)
                    logger.info(
                        "[STORE] Message updated: message_id=%s, has_subgraph_data=%s, has_sources=%s",
                        existing.id if existing else "unknown",
                        existing.subgraph_data is not None if existing else False,
                        existing.sources is not None if existing else False
                    )
                
                # 确保任务标记为完成
                if not getattr(latest, "done", False):
                    await mark_task_completed(
                        latest,
                        session_id,
                        user_message.id,
                        assistant_message_id=existing.id,
                    )
                return existing, False
        
        # 如果不存在，创建新消息（此时 subgraph_data 已存在或 return_subgraph=False）
        msg = await create_assistant_message(
            session_id,
            assistant_response,
            sources_for_frontend,
            subgraph_data,
            return_subgraph,
            raw_llm_response,
            raw_mindmap_response,
            deepsearch_trace_file_path=deepsearch_trace_file_path,
        )
        await mark_task_completed(
            latest,
            session_id,
            user_message.id,
            assistant_message_id=msg.id,
        )
        return msg, True


async def _ensure_finalization(
    assistant_response: str,
    chunks: list,
    subgraph_data: Optional[Any],
    subgraph_info: Optional[Any],
    raw_llm_response: Optional[Any],
    raw_mindmap_response: Optional[Any],
    enable_deepsearch: bool,
    deepsearch_result: Optional[Any],
    deepsearch_sources_for_frontend: Optional[Any],
    deepsearch_citation_key_map: Optional[dict],
    return_subgraph: bool,
    query: str,
    session_id: Any,
    first_turn: bool,
    include_evidence: bool,
    task_info: Optional[Any],
    user_message: Any,
    rag_inference_handler: Any,
    deepsearch_trace_file_path: Optional[str],
    citation_key_map_override: Optional[dict[int, int]] = None,
    assistant_response_renumbered: bool = False,
) -> None:
    """确保最终化逻辑执行（不yield事件，用于客户端断开时）"""
    # 检查取消
    if await check_and_handle_cancellation(task_info, session_id, user_message.id):
        return
    
    # 确保响应非空
    assistant_response, _ = ensure_non_empty_response(assistant_response)
    
    # 如果 return_subgraph=True 且 subgraph_data 为 None，需要生成 mindmap
    # 这与 _build_and_yield_final_response 中的逻辑保持一致，避免重复创建消息
    if return_subgraph and subgraph_data is None:
        logger.info(
            "[STORE] _ensure_finalization: Generating mindmap before creating message"
        )
        subgraph_data, raw_mindmap_response = await generate_mindmap_if_needed(
            return_subgraph,
            query,
            assistant_response,
            chunks,
            rag_inference_handler
        )
        logger.info(
            "[STORE] _ensure_finalization: Mindmap generation result: has_subgraph_data=%s",
            subgraph_data is not None
        )
    
    # 构建 sources 和 evidence
    if enable_deepsearch and deepsearch_result and deepsearch_sources_for_frontend is not None:
        sources_for_frontend = deepsearch_sources_for_frontend
        citation_key_map = deepsearch_citation_key_map
        if citation_key_map_override:
            from api.routers.chatbot import _renumber_sources_by_key_map

            sources_for_frontend = _renumber_sources_by_key_map(
                sources_for_frontend,
                citation_key_map_override,
            )
            citation_key_map = citation_key_map_override
    else:
        assistant_response, sources_for_frontend, citation_key_map = await build_sources_and_evidence(
            chunks,
            subgraph_data,
            subgraph_info,
            rag_inference_handler,
            assistant_response,
            False,
            citation_key_map=citation_key_map_override,
            assistant_response_renumbered=assistant_response_renumbered,
        )
    
    # Create the assistant message exactly once per task_id.
    # subgraph_data 已经在 create_assistant_message 中正确存储到 subgraph_data 字段
    deepsearch_trace = deepsearch_trace_file_path if enable_deepsearch else None
    assistant_message, created_now = await _get_or_create_final_assistant_message(
        session_id=session_id,
        assistant_response=assistant_response,
        sources_for_frontend=sources_for_frontend or [],
        subgraph_data=subgraph_data,
        return_subgraph=return_subgraph,
        raw_llm_response=raw_llm_response,
        raw_mindmap_response=raw_mindmap_response,
        task_info=task_info,
        user_message=user_message,
        deepsearch_trace_file_path=deepsearch_trace,
    )
    
    if task_info and created_now:
        logger.info(
            "Task %s finalization ensured: assistant_message_id=%s",
            task_info.task_id,
            assistant_message.id,
        )


async def _build_and_yield_final_response(
    assistant_response: str,
    chunks: list,
    subgraph_data: Optional[Any],
    subgraph_info: Optional[Any],
    raw_llm_response: Optional[Any],
    raw_mindmap_response: Optional[Any],
    enable_deepsearch: bool,
    deepsearch_result: Optional[Any],
    deepsearch_sources_for_frontend: Optional[Any],
    deepsearch_citation_key_map: Optional[dict],
    return_subgraph: bool,
    query: str,
    session_id: Any,
    first_turn: bool,
    include_evidence: bool,
    task_info: Optional[Any],
    user_message: Any,
    rag_inference_handler: Any,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    deepsearch_trace_file_path: Optional[str],
    emit_final_text: bool = True,
    citation_key_map_override: Optional[dict[int, int]] = None,
    assistant_response_renumbered: bool = False,
) -> AsyncGenerator[str, None]:
    """构建并生成最终响应事件"""
    # 检查取消（在生成 mindmap 前）
    if await check_and_handle_cancellation(task_info, session_id, user_message.id):
        async for cancel_event in yield_cancellation_event(request_id):
            yield cancel_event
        return
    
    # 生成 mindmap（如果需要）
    logger.info(
        "[STORE] Mindmap generation check: return_subgraph=%s, subgraph_data is None=%s, chunks_count=%d",
        return_subgraph,
        subgraph_data is None,
        len(chunks) if chunks else 0
    )
    if return_subgraph and subgraph_data is None:
        logger.info(
            "[STORE] Generating mindmap: return_subgraph=%s, subgraph_data_is_none=%s, chunks_count=%d",
            return_subgraph,
            subgraph_data is None,
            len(chunks) if chunks else 0
        )
        subgraph_data, raw_mindmap_response = await generate_mindmap_if_needed(
            return_subgraph,
            query,
            assistant_response,
            chunks,
            rag_inference_handler
        )
        logger.info(
            "[STORE] Mindmap generation result: has_subgraph_data=%s, nodes=%d, edges=%d",
            subgraph_data is not None,
            len(subgraph_data.get("nodes", [])) if subgraph_data else 0,
            len(subgraph_data.get("edges", [])) if subgraph_data else 0
        )
    elif not return_subgraph:
        logger.warning(
            "[STORE] Mindmap NOT generated: return_subgraph=False, subgraph_data will be None"
        )
    elif subgraph_data is not None:
        logger.info(
            "[STORE] Mindmap already exists: nodes=%d, edges=%d",
            len(subgraph_data.get("nodes", [])) if isinstance(subgraph_data, dict) else 0,
            len(subgraph_data.get("edges", [])) if isinstance(subgraph_data, dict) else 0
        )
    
    # 确保响应非空
    assistant_response, is_fallback_response = ensure_non_empty_response(assistant_response)
    
    # 构建 sources 和 evidence
    if enable_deepsearch and deepsearch_result and deepsearch_sources_for_frontend is not None:
        sources_for_frontend = deepsearch_sources_for_frontend
        citation_key_map = deepsearch_citation_key_map
        if citation_key_map_override:
            from api.routers.chatbot import _renumber_sources_by_key_map

            sources_for_frontend = _renumber_sources_by_key_map(
                sources_for_frontend,
                citation_key_map_override,
            )
            citation_key_map = citation_key_map_override
    else:
        assistant_response, sources_for_frontend, citation_key_map = await build_sources_and_evidence(
            chunks,
            subgraph_data,
            subgraph_info,
            rag_inference_handler,
            assistant_response,
            is_fallback_response,
            citation_key_map=citation_key_map_override,
            assistant_response_renumbered=assistant_response_renumbered,
        )
    
    # 检查取消（在创建消息前）
    if await check_and_handle_cancellation(task_info, session_id, user_message.id):
        async for cancel_event in yield_cancellation_event(request_id):
            yield cancel_event
        return

    # Emit a "final_text" event when streaming did not already normalize citations.
    final_text_payload = {
        "type": "final_text",
        "content": assistant_response,
        "citation_key_map": {str(k): v for k, v in (citation_key_map or {}).items()},
        "id": str(session_id),
    }
    if task_info:
        # Cache for task-resume so reconnects can always receive the normalized text.
        try:
            registry = get_chat_task_registry()
            await registry.append_event(task_info.task_id, final_text_payload, event_type="final_text")
        except Exception:  # noqa: BLE001
            pass
    if emit_final_text:
        yield sse_json_wrapped(final_text_payload, request_id=request_id)
    
    # Create the assistant message exactly once per task_id (even if the background callback runs).
    deepsearch_trace = deepsearch_trace_file_path if enable_deepsearch else None
    logger.info(
        "[STORE] Creating assistant_message: session_id=%s, return_subgraph=%s, has_subgraph_data=%s, subgraph_data_type=%s, sources_count=%d, chunks_count=%d",
        session_id,
        return_subgraph,
        subgraph_data is not None,
        type(subgraph_data).__name__ if subgraph_data is not None else "None",
        len(sources_for_frontend) if sources_for_frontend else 0,
        len(chunks) if chunks else 0
    )
    if subgraph_data:
        logger.info(
            "[STORE] Subgraph_data details before storage: nodes=%d, edges=%d, keys=%s",
            len(subgraph_data.get("nodes", [])) if isinstance(subgraph_data, dict) else 0,
            len(subgraph_data.get("edges", [])) if isinstance(subgraph_data, dict) else 0,
            list(subgraph_data.keys())[:10] if isinstance(subgraph_data, dict) else "N/A"
        )
    assistant_message, _created_now = await _get_or_create_final_assistant_message(
        session_id=session_id,
        assistant_response=assistant_response,
        sources_for_frontend=sources_for_frontend,
        subgraph_data=subgraph_data,
        return_subgraph=return_subgraph,
        raw_llm_response=raw_llm_response,
        raw_mindmap_response=raw_mindmap_response,
        task_info=task_info,
        user_message=user_message,
        deepsearch_trace_file_path=deepsearch_trace,
    )
    logger.info(
        "[STORE] Assistant_message created: id=%s, created_now=%s, stored_subgraph_data=%s, stored_sources=%s, stored_source_file_ids=%s",
        assistant_message.id,
        _created_now,
        assistant_message.subgraph_data is not None,
        assistant_message.sources is not None,
        assistant_message.source_file_ids is not None
    )
    if assistant_message.subgraph_data:
        logger.info(
            "[STORE] Stored subgraph_data verified: nodes=%d, edges=%d",
            len(assistant_message.subgraph_data.get("nodes", [])) if isinstance(assistant_message.subgraph_data, dict) else 0,
            len(assistant_message.subgraph_data.get("edges", [])) if isinstance(assistant_message.subgraph_data, dict) else 0
        )
    
    # 生成标题（如果是第一轮）
    if first_turn:
        title = await generate_and_update_title(
            first_turn,
            session_id,
            query,
            assistant_response,
            rag_inference_handler
        )
        if title:
            from ..event.event_generator import _yield_title_event
            async for event in _yield_title_event(title, request_id):
                yield event
    
    # 发送 sources 事件
    from ..event.event_generator import _yield_sources_event
    async for event in _yield_sources_event(
        sources_for_frontend,
        citation_key_map,
        session_id,
        request_id,
        task_info=task_info
    ):
        yield event
    
    # 构建 payload
    evidence = build_evidence_for_payload(
        include_evidence,
        chunks,
        subgraph_data,
        subgraph_info,
        rag_inference_handler
    )
    
    payload, subgraph_for_outer = build_stream_chat_payload(
        assistant_message,
        chunks,
        subgraph=subgraph_data if return_subgraph else None,
        evidence=evidence,
    )
    
    # 构建最终的 chunk 事件
    chunk_data = openai_chat_completion_chunk(
        chunk_id=chunk_id,
        model=model_name,
        created=created,
        delta=delta_envelope(role=None, tool_calls=[{
            "index": 0,
            "id": f"call_{assistant_message.id}",
            "type": "function",
            "function": {
                "name": "rag_arc_payload",
                "arguments": json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":")
                ),
            },
        }]),
    )
    
    if subgraph_for_outer is not None:
        chunk_data["subgraph"] = subgraph_for_outer

    if task_info:
        # Cache for task-resume so reconnects can receive the final payload consistently.
        try:
            registry = get_chat_task_registry()
            await registry.append_event(task_info.task_id, chunk_data, event_type="payload_chunk")
        except Exception:  # noqa: BLE001
            pass

    yield sse_json_wrapped(chunk_data, request_id=request_id)
    
    # 发送完成事件
    from ..event.event_generator import _yield_finish_event
    async for event in _yield_finish_event(chunk_id, model_name, created, request_id):
        yield event
