"""Response building for stream chat."""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Any, List
from encapsulation.data_model.orm_models import ChatMessage
from encapsulation.data_model.schema import Chunk
from encapsulation.data_model.deepsearch import EvidenceChunk
from api.routers.rag_inference_handlers import (
    get_rag_inference_handler,
    get_message_handler,
    get_session_handler,
    generate_title_via_llm,
)
from api.routers.chatbot import (
    _build_sources_for_frontend,
    _filter_and_renumber_sources_by_sup_keys_sorted,
)
from core.presentation.evidence import build_chat_evidence
from config.output_limits import CHAT_TOP_CHUNKS
from framework.thread_pool import get_thread_pool

logger = logging.getLogger(__name__)


def convert_evidence_chunks_to_chunks(evidence_chunks: List[EvidenceChunk | dict]) -> List[Chunk]:
    """Convert DeepSearch EvidenceChunk list (or dict list) to Chunk list."""
    chunks = []
    for evidence in evidence_chunks:
        # 处理字典格式的 evidence
        if isinstance(evidence, dict):
            chunk = Chunk(
                content=str(evidence.get("content") or ""),
                id=str(evidence.get("chunk_id") or ""),
                metadata={
                    "source": str(evidence.get("source") or ""),
                    "score": evidence.get("score"),
                    **(evidence.get("provenance") or {}),
                }
            )
        # 处理 EvidenceChunk 对象格式
        elif isinstance(evidence, EvidenceChunk):
            chunk = Chunk(
                content=evidence.content,
                id=evidence.chunk_id,
                metadata={
                    "source": evidence.source,
                    "score": evidence.score,
                    **evidence.provenance,
                }
            )
        else:
            # 未知格式，跳过或转换为字符串
            logger.warning("Unknown evidence format: %s, skipping", type(evidence).__name__)
            continue
        chunks.append(chunk)
    return chunks


async def generate_mindmap_if_needed(
    return_subgraph: bool,
    query: str,
    assistant_response: str,
    chunks: list,
    rag_inference_handler: Any
) -> tuple[Any, Any]:
    """Generate mindmap if return_subgraph is True."""
    if not return_subgraph:
        return None, None
    
    try:
        subgraph_data, raw_mindmap_response = await get_thread_pool().run_blocking(
            rag_inference_handler._generate_mindmap,
            query,
            assistant_response,
            chunks,
        )
        logger.info(
            "SSE generated mindmap: %d nodes, %d edges",
            len(subgraph_data.get("nodes", [])) if subgraph_data else 0,
            len(subgraph_data.get("edges", [])) if subgraph_data else 0
        )
        return subgraph_data, raw_mindmap_response
    except Exception as exc:
        logger.warning("Failed to generate mindmap in SSE: %s", exc)
        return None, None


def ensure_non_empty_response(assistant_response: str) -> tuple[str, bool]:
    """Ensure response is not empty, return fallback if needed."""
    if not assistant_response or not assistant_response.strip():
        logger.warning("Assistant response is empty; using fallback message")
        return "当前没有找到与您问题相关的内容，请尝试换个问法或提供更多信息。", True
    return assistant_response, False


async def build_sources_and_evidence(
    chunks: list,
    subgraph_data: Any,
    subgraph_info: Any,
    rag_inference_handler: Any,
    assistant_response: str,
    is_fallback_response: bool
) -> tuple[list, dict[int, int]]:
    """Build sources and evidence for response."""
    max_sources = int(os.getenv("CHATBOT_TOP_SOURCES", "5"))
    
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
    
    sources_for_frontend = await get_thread_pool().run_blocking(
        _build_sources_for_frontend,
        evidence_chunks,
        min(max_sources, CHAT_TOP_CHUNKS),
    )
    logger.info("SSE built %d sources for frontend", len(sources_for_frontend))
    
    citation_key_map: dict[int, int] = {}
    if not is_fallback_response:
        assistant_response, sources_for_frontend, citation_key_map = (
            _filter_and_renumber_sources_by_sup_keys_sorted(
                sources_for_frontend,
                assistant_response,
            )
        )
        logger.info(
            "SSE citation normalization applied: cited_sources=%d citation_key_map=%s",
            len(sources_for_frontend),
            citation_key_map,
        )
    
    return sources_for_frontend, citation_key_map


async def create_assistant_message(
    session_id: uuid.UUID,
    assistant_response: str,
    sources_for_frontend: list,
    subgraph_data: Any,
    return_subgraph: bool,
    raw_llm_response: Any,
    raw_mindmap_response: Any
) -> ChatMessage:
    """Create and save assistant message."""
    message_handler = get_message_handler()
    sources_for_storage = (
        [s.model_dump() for s in sources_for_frontend]
        if sources_for_frontend
        else None
    )
    
    assistant_message = ChatMessage(
        session_id=session_id,
        content={"role": "assistant", "content": assistant_response},
        source_file_ids=(
            [s.chunk_id for s in sources_for_frontend]
            if sources_for_frontend
            else None
        ),
        sources=sources_for_storage,
        subgraph_data=subgraph_data if return_subgraph else None,
        raw_llm_response=raw_llm_response,
        raw_mindmap_response=(
            {"response": raw_mindmap_response}
            if raw_mindmap_response
            else None
        ),
        created_at=datetime.now(),
    )
    return await get_thread_pool().run_blocking(
        message_handler.create_message,
        assistant_message
    )


async def generate_and_update_title(
    first_turn: bool,
    session_id: uuid.UUID,
    query: str,
    assistant_response: str,
    rag_inference_handler: Any
) -> str | None:
    """Generate and update session title if first turn."""
    if not first_turn:
        return None
    
    try:
        title = await generate_title_via_llm(
            rag_inference_handler,
            query.strip(),
            assistant_response.strip(),
        )
        await get_thread_pool().run_blocking(
            get_session_handler().update_session,
            session_id,
            {"name": title},
        )
        logger.info("SSE generated and updated session title: session_id=%s, title=%s", session_id, title)
        return title
    except Exception as exc:
        logger.warning("Failed to generate title: %s", exc)
        return None


def build_evidence_for_payload(
    include_evidence: bool,
    chunks: list,
    subgraph_data: Any,
    subgraph_info: Any,
    rag_inference_handler: Any
) -> Any:
    """Build evidence for payload if requested."""
    if not include_evidence:
        return None
    
    graph_store = None
    try:
        graph_store = rag_inference_handler.get_graph_store()
    except Exception:  # noqa: BLE001
        graph_store = None
    
    return build_chat_evidence(
        chunks,
        subgraph_data=subgraph_data,
        subgraph_info=subgraph_info,
        max_chunks=CHAT_TOP_CHUNKS,
        graph_store=graph_store,
    )
