"""DeepSearch 处理模块"""
import logging
from typing import Any, Optional, AsyncGenerator
from .deepsearch_handler import process_deepsearch
from .deepsearch_handler_ext import process_deepsearch_result
from ..response.response_builder import convert_evidence_chunks_to_chunks
from ..task.task_helpers import check_and_handle_cancellation, yield_cancellation_event, cache_deepsearch_event
from ..task.task_registry import ChatTaskInfo
from ..event.event_generator import _yield_deepsearch_answer_stream

logger = logging.getLogger(__name__)


async def process_deepsearch_with_events(
    query: str,
    effective_owner: Any,
    request_id: str,
    chunk_id: str,
    model_name: str,
    created: int,
    loop: Any,
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any,
    response_parts: list[str]
) -> AsyncGenerator[Any, None]:
    """处理 DeepSearch 并生成事件流，最后 yield 结果元组"""
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
        if await check_and_handle_cancellation(task_info, session_id, user_message_id):
            async for cancel_event in yield_cancellation_event(request_id):
                yield cancel_event
            return
        
        await cache_deepsearch_event(task_info, event)
        yield event
    
    # 获取结果
    deepsearch_result = deepsearch_result_container[0]
    deepsearch_trace_file_path = trace_file_path_container[0]
    
    if not deepsearch_result:
        logger.warning("DeepSearch failed or returned None, falling back to RAG system")
        yield (None, None, None, None, [], "")
        return
    
    # 处理 DeepSearch 结果
    deepsearch_answer, deepsearch_evidences, deepsearch_sources_for_frontend, deepsearch_citation_key_map = await process_deepsearch_result(
        deepsearch_result,
        False  # include_evidence
    )
    
    if not deepsearch_answer:
        logger.warning("DeepSearch completed but no answer found, falling back to RAG system")
        yield (None, None, None, None, [], "")
        return
    
    logger.info("DeepSearch completed, using DeepSearch answer directly (length=%d, type=%s)", 
              len(deepsearch_answer), type(deepsearch_answer).__name__)
    
    # 流式输出 DeepSearch 答案
    async for event in _yield_deepsearch_answer_stream(
        deepsearch_answer,
        chunk_id,
        model_name,
        created,
        request_id,
        response_parts,
        task_info
    ):
        yield event
    
    # 转换 evidences 为 chunks
    chunks = convert_evidence_chunks_to_chunks(deepsearch_evidences)
    assistant_response = "".join(response_parts)
    
    # 最后 yield 结果元组
    yield (deepsearch_result, deepsearch_trace_file_path, deepsearch_sources_for_frontend, 
           deepsearch_citation_key_map, chunks, assistant_response)
