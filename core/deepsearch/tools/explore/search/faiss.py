"""FAISS search channel and tool."""
import asyncio
from typing import Any, Dict, List

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from core.deepsearch.utils.file_scope import resolve_file_scope
from framework.thread_pool import get_thread_pool

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_PRIMARY, SCOPE_FILE, SCOPE_OWNER
from .base import _ChannelResult, _SearchToolBase, strip_file_scope_from_graph_context


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

        visibility = self._resolve_owner_visibility(request)
        if not visibility.enabled:
            return _ChannelResult(
                channel="faiss",
                evidences=[],
                diagnostics={"query": query, "reason": "owner_id_missing", "owner_visibility": visibility.as_dict()},
                summary="FAISS search skipped: missing owner scope.",
            )

        override = request.extra.get("faiss_top_k")
        effective_top_k = self._resolve_top_k(override, top_k)

        async def _call_one(owner_id: str) -> List[Chunk]:
            return await get_thread_pool().run_blocking(
                retrievers.dense.invoke,
                query,
                k=effective_top_k,
                owner_id=owner_id,
                with_score=True,
            )

        owner_ids = list(visibility.owner_ids_used)
        parts = await asyncio.gather(*[_call_one(owner_id) for owner_id in owner_ids], return_exceptions=True)
        chunks: List[Chunk] = []
        per_owner: Dict[str, int] = {}
        errors: List[str] = []
        for owner_id, part in zip(owner_ids, parts):
            if isinstance(part, Exception):
                errors.append(f"{owner_id}: {part}")
                continue
            per_owner[owner_id] = len(part)
            chunks.extend(part)
        chunks, dropped = self._apply_file_scope(chunks, file_scope)
        section_scope = self._resolve_section_scope(request.extra or {})
        chunks, section_dropped = self._apply_section_scope(chunks, section_scope)

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
                    "owner_id": getattr(chunk, "owner_id", None) or meta.get("owner_id"),
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
            "section_scope": sorted(section_scope),
            "section_scope_dropped": section_dropped,
            "owner_visibility": visibility.as_dict(),
            "per_owner_retrieved": per_owner,
            "errors": errors,
            "results": results,
        }
        return _ChannelResult(channel="faiss", evidences=evidences, diagnostics=diagnostics, summary=summary)


class SearchFaissTool(_SearchToolBase, _FaissChannel, GraphTool):
    """FAISS-only scoped search tool (requires file scope)."""

    descriptor = ToolDescriptor(
        name="search.scoped.faiss",
        channel="graph",
        description="FAISS-only dense search scoped to a file_id/file_ids (prevents cross-document noise).",
        speed="fast",
        cost="low",
        strategy_tags=("search", "faiss", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.search.scoped.faiss",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "file_id": {"type": "string", "description": "Restrict results to a specific file_id (required unless file_ids provided)."},
                "file_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to file_ids (required unless file_id provided)."},
                "filename_contains": {"type": "array", "items": {"type": "string"}, "description": "Best-effort filename filter."},
                "section_id": {"type": "string", "description": "Restrict results to a specific section_id."},
                "section_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to section_ids."},
                "owner_ids": {"type": "array", "items": {"type": "string"}, "description": "Owner ids to search (me/share)."},
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
        if not getattr(file_scope, "file_ids", None):
            return ToolResult(
                summary="search.scoped.faiss skipped: missing file_id/file_ids (call search.file first).",
                diagnostics={"reason": "missing_file_scope", "query": query},
            )
        top_k = self._resolve_top_k(request.extra.get("top_k"), tool_defaults.SEARCH_DEFAULT_TOP_K)
        result = await self._run_faiss(request=request, query=query, top_k=top_k, file_scope=file_scope)
        return ToolResult(summary=result.summary, evidences=result.evidences, diagnostics=result.diagnostics)


class SearchGlobalFaissTool(_SearchToolBase, _FaissChannel, GraphTool):
    """FAISS-only global search tool (does not inherit file_scope)."""

    descriptor = ToolDescriptor(
        name="search.global.faiss",
        channel="graph",
        description=(
            "FAISS-only dense search across all accessible documents. "
            "Warning: may introduce cross-document/company noise. Prefer search.scoped.faiss."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "faiss", "global", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.search.global.faiss",
        mcp_callable=True,
        input_schema=SearchFaissTool.descriptor.input_schema,
        example_args={
            "question": "Find semantic matches globally",
            "plan_step": "plan_01",
            "extra": {"channels": ["faiss"], "top_k": 10},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)
        patched_ctx = strip_file_scope_from_graph_context(request.graph_context)
        file_scope = resolve_file_scope(
            extra=request.extra,
            graph_context_metadata=(patched_ctx.metadata if patched_ctx else {}),
            question=request.question,
        )
        top_k = self._resolve_top_k(request.extra.get("top_k"), tool_defaults.SEARCH_DEFAULT_TOP_K)
        patched = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=dict(request.extra or {}),
            graph_context=patched_ctx,
            coverage_metrics=request.coverage_metrics,
        )
        result = await self._run_faiss(request=patched, query=query, top_k=top_k, file_scope=file_scope)
        risk = "global_search_may_introduce_cross_doc_noise"
        diagnostics = {**dict(result.diagnostics or {}), "search_mode": "global", "risk": risk}
        summary = (result.summary or "FAISS search completed.").rstrip()
        summary = summary + f" NOTE: {risk}."
        return ToolResult(summary=summary, evidences=result.evidences, diagnostics=diagnostics)
