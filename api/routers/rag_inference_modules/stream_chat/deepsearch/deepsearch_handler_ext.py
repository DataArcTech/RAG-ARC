"""DeepSearch 处理辅助函数"""
import json
import logging
from typing import Any, AsyncGenerator, Optional, Tuple
from core.presentation.deepsearch_payload import trim_deepsearch_payload
from ..task.task_helpers import check_and_handle_cancellation, cache_deepsearch_event, yield_cancellation_event
from ..task.task_registry import ChatTaskInfo
from ..response.response_builder import build_deepsearch_sources_for_frontend, convert_evidence_chunks_to_chunks
from .deepsearch_handler import process_deepsearch
from api.sse import sse_json_wrapped

logger = logging.getLogger(__name__)


async def process_deepsearch_with_cancellation_check(
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
) -> Tuple[Optional[Any], Optional[str], AsyncGenerator[str, None]]:
    """处理 DeepSearch 并检查取消状态"""
    deepsearch_result_container, trace_file_path_container, deepsearch_gen = await process_deepsearch(
        query,
        str(effective_owner),
        request_id,
        chunk_id,
        model_name,
        created,
        loop
    )
    
    # 处理 DeepSearch 事件流
    async for event in deepsearch_gen:
        # 检查任务是否被取消
        if await check_and_handle_cancellation(task_info, session_id, user_message_id):
            async for cancel_event in yield_cancellation_event(request_id):
                yield cancel_event
            return None, None, None
        
        # 缓存事件
        await cache_deepsearch_event(task_info, event)
        yield event
    
    # 获取结果
    deepsearch_result = deepsearch_result_container[0]
    deepsearch_trace_file_path = trace_file_path_container[0]
    
    return deepsearch_result, deepsearch_trace_file_path, None


def extract_deepsearch_answer(report: dict) -> str:
    """从 DeepSearch report 中提取答案"""
    raw_answer = report.get("answer") or ""
    
    if not isinstance(raw_answer, str):
        if isinstance(raw_answer, dict):
            raw_answer = (
                raw_answer.get("text") or 
                raw_answer.get("content") or 
                raw_answer.get("short_answer") or 
                str(raw_answer)
            )
        else:
            raw_answer = str(raw_answer)
    
    return raw_answer.strip() if raw_answer else ""


async def process_deepsearch_result(
    deepsearch_result: Any,
    include_evidence: bool
) -> Tuple[Optional[str], list, Optional[Any], Optional[dict]]:
    """处理 DeepSearch 结果并提取信息"""
    if not deepsearch_result:
        return None, [], None, None
    
    trimmed = trim_deepsearch_payload(deepsearch_result, include_evidence=False)
    report = trimmed.get("report") if isinstance(trimmed, dict) else {}
    if not isinstance(report, dict):
        report = {}
    
    deepsearch_answer = extract_deepsearch_answer(report)
    deepsearch_evidences = report.get("evidences") or []
    deepsearch_sources_for_frontend, citation_key_map = build_deepsearch_sources_for_frontend(report)
    
    return deepsearch_answer, deepsearch_evidences, deepsearch_sources_for_frontend, citation_key_map
