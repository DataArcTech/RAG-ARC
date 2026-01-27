"""事件生成器辅助函数"""
import logging
from collections.abc import AsyncGenerator
from typing import Any, Optional, Tuple

from ..task.task_helpers import check_and_handle_cancellation, yield_cancellation_event, mark_task_completed
from ..deepsearch.deepsearch_handler import process_deepsearch
from ..deepsearch.deepsearch_handler_ext import process_deepsearch_result
from ..response.response_builder import convert_evidence_chunks_to_chunks
from ..rag.stream_processor import start_stream_processing
from ..task.task_registry import ChatTaskInfo

logger = logging.getLogger(__name__)


async def process_deepsearch_events(
    query: str,
    effective_owner: Any,
    request_id: str,
    chunk_id: str,
    model_name: str,
    created: int,
    loop: Any,
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any
) -> AsyncGenerator[Any, None]:
    """处理 DeepSearch 事件流。

    Note: this is an async generator (yields events). The final DeepSearch result is
    stored in the containers returned by `process_deepsearch()` for the caller to read.
    """
    deepsearch_result_container, trace_file_path_container, deepsearch_gen = await process_deepsearch(
        query,
        str(effective_owner),
        request_id,
        chunk_id,
        model_name,
        created,
        loop
    )
    
    # 处理事件流
    async for event in deepsearch_gen:
        # 检查取消
        if await check_and_handle_cancellation(task_info, session_id, user_message_id):
            return

        # 缓存事件
        from .task_helpers import cache_deepsearch_event
        await cache_deepsearch_event(task_info, event)
        yield event
    
    # Stream exhausted. Caller can inspect deepsearch_result_container/trace_file_path_container.
    _ = deepsearch_result_container[0]
    _ = trace_file_path_container[0]


async def process_rag_streaming(
    query: str,
    effective_owner: Any,
    return_subgraph: bool,
    include_evidence: bool,
    history_text: Optional[str],
    enable_web_search: bool,
    queue: Any,
    loop: Any,
    prepared: dict,
    stream_error: list,
    emit_progress: Any,
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any
) -> bool:
    """处理 RAG 流式处理，返回是否成功（False 表示被取消）"""
    # 检查取消
    if await check_and_handle_cancellation(task_info, session_id, user_message_id):
        return False
    
    # 启动流处理
    start_stream_processing(
        query,
        effective_owner,
        return_subgraph,
        include_evidence,
        history_text,
        enable_web_search,
        queue,
        loop,
        prepared,
        stream_error,
        emit_progress
    )
    
    return True


async def build_final_response(
    response_parts: list[str],
    prepared: dict[str, Any],
    enable_deepsearch: bool,
    deepsearch_result: Optional[Any],
    deepsearch_sources_for_frontend: Optional[Any],
    deepsearch_citation_key_map: Optional[dict],
    chunks: list,
    subgraph_data: Optional[Any],
    subgraph_info: Optional[Any],
    rag_inference_handler: Any
) -> Tuple[str, list, Optional[Any], Optional[Any], Optional[Any], Optional[Any], Any, dict]:
    """构建最终响应数据"""
    assistant_response = "".join(response_parts)
    
    if not enable_deepsearch or not deepsearch_result:
        # 从 prepared 获取数据
        chunks = prepared.get("chunks") or []
        subgraph_data = prepared.get("subgraph_data")
        subgraph_info = prepared.get("subgraph_info")
        raw_llm_response = prepared.get("raw_llm_response")
        raw_mindmap_response = prepared.get("raw_mindmap_response")
    else:
        # DeepSearch 已处理
        raw_llm_response = None
        raw_mindmap_response = None
    
    return assistant_response, chunks, subgraph_data, subgraph_info, raw_llm_response, raw_mindmap_response, None, {}
