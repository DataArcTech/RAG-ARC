"""FAISS search channel and tool."""
import asyncio
from typing import Any, Dict, List

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from core.deepsearch.utils.file_scope import resolve_file_scope

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_PRIMARY, SCOPE_FILE, SCOPE_OWNER
from .base import _ChannelResult, _SearchToolBase


class _FaissChannel:
    async def _run_faiss(
        self: _SearchToolBase,
        *,
        request: ToolRunRequest,
        query: str,
        top_k: int,
        file_scope,
    ) -> _ChannelResult:
        retrievers = self._resolve_retrievers()
        if retrievers.dense is None:
            return _ChannelResult(
                channel="faiss",
                evidences=[],
                diagnostics={"query": query, "reason": "dense_retriever_unavailable"},
                summary="FAISS search skipped: dense retriever unavailable.",
            )

        owner_id = self._resolve_owner_id(request)
        if owner_id is None:
            return _ChannelResult(
                channel="faiss",
                evidences=[],
                diagnostics={"query": query, "reason": "owner_id_missing"},
                summary="FAISS search skipped: missing owner scope.",
            )

        override = request.extra.get("faiss_top_k")
        effective_top_k = self._resolve_top_k(override, top_k)

        def _call() -> List[Chunk]:
            return retrievers.dense.invoke(query, k=effective_top_k, owner_id=owner_id, with_score=True)

        chunks = await asyncio.to_thread(_call)
        chunks, dropped = self._apply_file_scope(chunks, file_scope)

        evidences: List[EvidenceChunk] = []
        results: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[:effective_top_k]):
            content = self._chunk_content(chunk)
            meta = self._chunk_meta(chunk)
            snippet = self._summary_window(content)
            chunk_id = self._chunk_id(chunk, "faiss")
            score = self._chunk_score(chunk)
            file_name = self._chunk_file_name(meta)
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source="faiss",
                content=snippet,
                kind=EVIDENCE_KIND_PRIMARY,
                score=score,
                provenance={
                    "channel": "faiss",
                    "rank": idx,
                    "file_name": file_name,
                    "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
                    "metadata": meta,
                },
            )
            evidences.append(evidence)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "file_name": file_name,
                    "summary": snippet,
                }
            )

        summary = f"FAISS search returned {len(evidences)} chunks." if evidences else "FAISS search returned no chunks."
        diagnostics = {
            "query": query,
            "top_k": effective_top_k,
            "retrieved": len(chunks),
            "file_scope_dropped": dropped,
            "results": results,
        }
        return _ChannelResult(channel="faiss", evidences=evidences, diagnostics=diagnostics, summary=summary)


class SearchFaissTool(_SearchToolBase, _FaissChannel, GraphTool):
    """FAISS-only search tool."""

    descriptor = ToolDescriptor(
        name="search.faiss",
        channel="graph",
        description="FAISS-only dense search for quick semantic matches.",
        speed="fast",
        cost="low",
        strategy_tags=("search", "faiss", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.search.faiss",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "top_k": {"type": "integer", "minimum": 0, "description": "Top-k results to return."},
                "faiss_top_k": {"type": "integer", "minimum": 0, "description": "Alias of top_k."},
            }
        ),
        example_args={
            "question": "HippoRAG retrieval",
            "plan_step": "plan_01",
            "extra": {"top_k": 10},
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
        result = await self._run_faiss(request=request, query=query, top_k=top_k, file_scope=file_scope)
        return ToolResult(summary=result.summary, evidences=result.evidences, diagnostics=result.diagnostics)
