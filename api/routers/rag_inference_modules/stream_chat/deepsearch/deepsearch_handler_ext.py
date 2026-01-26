"""DeepSearch 处理辅助函数"""
import logging
from typing import Any, Optional, Tuple
from core.presentation.deepsearch_payload import trim_deepsearch_payload
from ..response.response_builder import build_deepsearch_sources_for_frontend

logger = logging.getLogger(__name__)


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
