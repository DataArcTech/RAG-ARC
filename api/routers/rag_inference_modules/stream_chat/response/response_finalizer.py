"""最终响应构建函数"""
import json
import logging
from typing import Any, Optional, AsyncGenerator
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

logger = logging.getLogger(__name__)


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
    deepsearch_trace_file_path: Optional[str]
) -> None:
    """确保最终化逻辑执行（不yield事件，用于客户端断开时）"""
    # 检查取消
    if await check_and_handle_cancellation(task_info, session_id, user_message.id):
        return
    
    # 确保响应非空
    assistant_response, _ = ensure_non_empty_response(assistant_response)
    
    # 构建 sources 和 evidence（简化版，不生成mindmap）
    if enable_deepsearch and deepsearch_result and deepsearch_sources_for_frontend is not None:
        sources_for_frontend = deepsearch_sources_for_frontend
    else:
        _, sources_for_frontend, _ = await build_sources_and_evidence(
            chunks,
            subgraph_data,
            subgraph_info,
            rag_inference_handler,
            assistant_response,
            False
        )
    
    # 创建 assistant message
    assistant_message = await create_assistant_message(
        session_id,
        assistant_response,
        sources_for_frontend or [],
        subgraph_data,
        return_subgraph,
        raw_llm_response,
        raw_mindmap_response,
        deepsearch_trace_file_path=deepsearch_trace_file_path if enable_deepsearch else None
    )
    
    # 任务完成，更新状态
    if task_info:
        try:
            await mark_task_completed(
                task_info,
                session_id,
                user_message.id,
                assistant_message_id=assistant_message.id
            )
            logger.info("Task %s finalization ensured: assistant_message_id=%s", 
                       task_info.task_id, assistant_message.id)
        except Exception as e:
            logger.warning("Failed to update task completion status: %s", e)


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
    deepsearch_trace_file_path: Optional[str]
) -> AsyncGenerator[str, None]:
    """构建并生成最终响应事件"""
    # 检查取消（在生成 mindmap 前）
    if await check_and_handle_cancellation(task_info, session_id, user_message.id):
        async for cancel_event in yield_cancellation_event(request_id):
            yield cancel_event
        return
    
    # 生成 mindmap（如果需要）
    if return_subgraph and subgraph_data is None:
        subgraph_data, raw_mindmap_response = await generate_mindmap_if_needed(
            return_subgraph,
            query,
            assistant_response,
            chunks,
            rag_inference_handler
        )
    
    # 确保响应非空
    assistant_response, is_fallback_response = ensure_non_empty_response(assistant_response)
    
    # 构建 sources 和 evidence
    if enable_deepsearch and deepsearch_result and deepsearch_sources_for_frontend is not None:
        sources_for_frontend = deepsearch_sources_for_frontend
        citation_key_map = deepsearch_citation_key_map
    else:
        assistant_response, sources_for_frontend, citation_key_map = await build_sources_and_evidence(
            chunks,
            subgraph_data,
            subgraph_info,
            rag_inference_handler,
            assistant_response,
            is_fallback_response
        )
    
    # 检查取消（在创建消息前）
    if await check_and_handle_cancellation(task_info, session_id, user_message.id):
        async for cancel_event in yield_cancellation_event(request_id):
            yield cancel_event
        return
    
    # 创建 assistant message
    assistant_message = await create_assistant_message(
        session_id,
        assistant_response,
        sources_for_frontend,
        subgraph_data,
        return_subgraph,
        raw_llm_response,
        raw_mindmap_response,
        deepsearch_trace_file_path=deepsearch_trace_file_path if enable_deepsearch else None
    )
    
    # 任务完成，更新状态
    if task_info:
        try:
            await mark_task_completed(
                task_info,
                session_id,
                user_message.id,
                assistant_message_id=assistant_message.id
            )
        except Exception as e:
            logger.warning("Failed to update task completion status: %s", e)
    
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
    
    yield sse_json_wrapped(chunk_data, request_id=request_id)
    
    # 发送完成事件
    from ..event.event_generator import _yield_finish_event
    async for event in _yield_finish_event(chunk_id, model_name, created, request_id):
        yield event
