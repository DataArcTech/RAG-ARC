"""Multi-channel search tool orchestrator."""
import asyncio
from typing import Any, Dict, List

from config.core.deepsearch import tool_defaults
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.file_scope import resolve_file_scope

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_PRIMARY, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from .base import _SearchToolBase
from .bm25 import _Bm25Channel
from .faiss import _FaissChannel
from .graph_chunk import _GraphChunkChannel


class SearchTool(_SearchToolBase, _FaissChannel, _Bm25Channel, _GraphChunkChannel, GraphTool):
    """Search across faiss, bm25, and graph_chunk channels."""

    descriptor = ToolDescriptor(
        name="search",
        channel="graph",
        description=(
            "Three-way fast search (faiss + bm25 + graph_chunk) for quick localization. "
            "Evidence: primary snippets (citeable); defaults to all channels but allows per-channel selection."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "faiss", "bm25", "graph_chunk", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.search",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Channels to execute: faiss, bm25, graph_chunk (default: all).",
                },
                "top_k": {"type": "integer", "minimum": 0, "description": "Default top-k for all channels."},
                "faiss_top_k": {"type": "integer", "minimum": 0, "description": "Top-k override for faiss."},
                "bm25_top_k": {"type": "integer", "minimum": 0, "description": "Top-k override for bm25."},
                "graph_top_k": {"type": "integer", "minimum": 0, "description": "Top-k override for graph_chunk."},
                "use_ppr": {"type": "boolean", "description": "Enable PPR for graph_chunk (default off)."},
                "enable_llm_rerank": {"type": "boolean", "description": "Enable LLM fact rerank for graph_chunk."},
                "enable_entity_fallback": {"type": "boolean", "description": "Enable entity fallback for graph_chunk."},
                "entity_seed_top_k": {"type": "integer", "minimum": 0, "description": "Max entities from fallback."},
                "seed_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional seed entities (entity_id or name).",
                },
                "bm25_use_phrase_query": {"type": "boolean", "description": "Use phrase query for BM25."},
                "graph_overrides": {
                    "type": "object",
                    "description": "Graph retrieval overrides (allowlisted safe keys only).",
                },
            }
        ),
        example_args={
            "question": "HippoRAG graph retrieval",
            "plan_step": "plan_01",
            "extra": {"channels": ["faiss", "bm25", "graph_chunk"], "top_k": 10},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)
        file_scope = resolve_file_scope(
            extra=request.extra,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
            question=request.question,
        )
        top_k = self._resolve_top_k(request.extra.get("top_k"), tool_defaults.SEARCH_DEFAULT_TOP_K)
        channels, unknown = self._resolve_channels(request.extra or {})
        if not channels:
            return ToolResult(
                summary="Search skipped: no channels selected.",
                diagnostics={"query": query, "channels": [], "unknown_channels": unknown},
            )

        tasks: List[asyncio.Task] = []
        for channel in channels:
            if channel == "faiss":
                tasks.append(asyncio.create_task(self._run_faiss(request=request, query=query, top_k=top_k, file_scope=file_scope)))
            elif channel == "bm25":
                tasks.append(asyncio.create_task(self._run_bm25(request=request, query=query, top_k=top_k, file_scope=file_scope)))
            elif channel == "graph_chunk":
                tasks.append(asyncio.create_task(self._run_graph_chunk(request=request, query=query, top_k=top_k, file_scope=file_scope)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors: List[str] = []
        evidences: List[EvidenceChunk] = []
        windows: Dict[str, Any] = {}
        summaries: List[str] = []
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue
            evidences.extend(result.evidences)
            windows[result.channel] = result.diagnostics
            summaries.append(result.summary)

        if not evidences and errors:
            raise RuntimeError(f"search failed: {errors[0]}")

        summary = "Search completed." + (" " + " ".join(summaries) if summaries else "")
        diagnostics = {
            "query": query,
            "channels": channels,
            "unknown_channels": unknown,
            "errors": errors,
            "windows": windows,
        }
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)
