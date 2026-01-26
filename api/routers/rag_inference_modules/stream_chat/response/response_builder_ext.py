"""响应构建辅助函数"""
import logging
import json
from typing import Any, Optional, Tuple
from .response_builder import (
    generate_mindmap_if_needed,
    build_sources_and_evidence,
    build_evidence_for_payload,
)
from api.routers.rag_inference_models import build_stream_chat_payload
from api.sse import (
    openai_chat_completion_chunk,
    delta_envelope,
    sse_json_wrapped,
)
from ..event.event_generator import _yield_finish_event

logger = logging.getLogger(__name__)


async def build_response_data(
    response_parts: list[str],
    prepared: dict[str, Any],
    enable_deepsearch: bool,
    deepsearch_result: Optional[Any],
    deepsearch_sources_for_frontend: Optional[Any],
    deepsearch_citation_key_map: Optional[dict],
    chunks: list,
    subgraph_data: Optional[Any],
    subgraph_info: Optional[Any],
    rag_inference_handler: Any,
    assistant_response: str,
    is_fallback_response: bool
) -> Tuple[str, Any, dict]:
    """构建响应数据：sources 和 citation_key_map"""
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
    
    return assistant_response, sources_for_frontend, citation_key_map


async def process_mindmap_if_needed(
    return_subgraph: bool,
    query: str,
    assistant_response: str,
    chunks: list,
    rag_inference_handler: Any,
    subgraph_data: Optional[Any]
) -> Tuple[Optional[Any], Optional[Any]]:
    """如果需要，生成 mindmap"""
    raw_mindmap_response = None
    if return_subgraph and subgraph_data is None:
        subgraph_data, raw_mindmap_response = await generate_mindmap_if_needed(
            return_subgraph,
            query,
            assistant_response,
            chunks,
            rag_inference_handler
        )
    return subgraph_data, raw_mindmap_response


async def build_final_response_payload(
    assistant_message: Any,
    chunks: list,
    subgraph_data: Optional[Any],
    subgraph_info: Optional[Any],
    return_subgraph: bool,
    include_evidence: bool,
    rag_inference_handler: Any
) -> Tuple[dict, Optional[Any]]:
    """构建最终的响应 payload"""
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
    
    return payload, subgraph_for_outer


async def yield_final_payload_event(
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    assistant_message_id: Any,
    payload: dict,
    subgraph_for_outer: Optional[Any]
) -> Any:
    """生成最终的 payload 事件"""
    chunk_data = openai_chat_completion_chunk(
        chunk_id=chunk_id,
        model=model_name,
        created=created,
        delta=delta_envelope(role=None, tool_calls=[{
            "index": 0,
            "id": f"call_{assistant_message_id}",
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
    
    async for event in _yield_finish_event(chunk_id, model_name, created, request_id):
        yield event
