"""RAG 流式处理模块"""
import logging
from typing import Any, Optional
from .stream_processor import start_stream_processing
from ..task.task_helpers import check_and_handle_cancellation, yield_cancellation_event, mark_task_completed
from ..event.event_generator import _process_queue_events_with_cache, _yield_error_event
from ..task.task_registry import ChatTaskInfo

logger = logging.getLogger(__name__)


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
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any
) -> bool:
    """
    启动 RAG 流式处理
    
    返回: success (False 表示被取消)
    """
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


async def process_rag_queue_events(
    queue: Any,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    stream_error: list,
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any,
    citation_stream: Any = None,
) -> Any:
    """处理 RAG 队列事件并检查取消"""
    async for event in _process_queue_events_with_cache(
        queue,
        chunk_id,
        model_name,
        created,
        request_id,
        response_parts,
        stream_error,
        task_info,
        citation_stream,
    ):
        # 检查取消
        if await check_and_handle_cancellation(task_info, session_id, user_message_id):
            async for cancel_event in yield_cancellation_event(request_id):
                yield cancel_event
            return
        yield event


async def handle_rag_streaming_error(
    stream_error: list,
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any,
    request_id: str
) -> Any:
    """处理 RAG 流式处理错误"""
    if stream_error[0] is not None:
        if task_info:
            try:
                error_msg = str(stream_error[0])
                await mark_task_completed(task_info, session_id, user_message_id, error=error_msg)
            except Exception as e:
                logger.warning("Failed to update task error status: %s", e)
        
        async for event in _yield_error_event(stream_error[0], request_id):
            yield event
