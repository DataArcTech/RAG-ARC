"""FAISS search channel and tool."""
import asyncio
from typing import Any, Dict, List

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from framework.thread_pool import get_thread_pool

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER
from .base import _ChannelResult, _SearchToolBase, resolve_explicit_file_scope, strip_file_scope_from_graph_context


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

        async def _call_one(owner_id: str, query_text: str) -> List[Chunk]:
            return await get_thread_pool().run_blocking(
                retrievers.dense.invoke,
                query_text,
                k=effective_top_k,
                owner_id=owner_id,
                with_score=True,
            )

        owner_ids = list(visibility.owner_ids_used)
        cache_scope = self._resolve_owner_id(request)
        query_variants = self._resolve_query_variants(query, cache_scope=cache_scope)
        parts = await asyncio.gather(
            *[_call_one(owner_id, qv) for qv in query_variants for owner_id in owner_ids],
            return_exceptions=True,
        )
        chunks: List[Chunk] = []
        per_owner: Dict[str, int] = {}
        per_variant: Dict[str, int] = {}
        errors: List[str] = []
        part_idx = 0
        for qv in query_variants:
            variant_count = 0
            for owner_id in owner_ids:
                if part_idx >= len(parts):
                    break
                part = parts[part_idx]
                part_idx += 1
                if isinstance(part, Exception):
                    errors.append(f"{owner_id}: {part}")
                    continue
                variant_count += len(part)
                per_owner[owner_id] = per_owner.get(owner_id, 0) + len(part)
                chunks.extend(part)
            per_variant[str(qv)] = variant_count
        # Deduplicate across variants by chunk_id, keep highest-score instance if available.
        deduped: Dict[str, Chunk] = {}
        deduped_scores: Dict[str, float] = {}
        for chunk in chunks:
            chunk_id = self._chunk_id(chunk, "faiss")
            score = self._chunk_score(chunk)
            if chunk_id not in deduped:
                deduped[chunk_id] = chunk
                if score is not None:
                    deduped_scores[chunk_id] = score
                continue
            if score is not None:
                prev = deduped_scores.get(chunk_id)
                if prev is None or score > prev:
                    deduped[chunk_id] = chunk
                    deduped_scores[chunk_id] = score
        chunks = list(deduped.values())
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
                kind=EVIDENCE_KIND_DERIVED,
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
            "query_variants": query_variants,
            "top_k": effective_top_k,
            "retrieved": len(chunks),
            "file_scope_dropped": dropped,
            "section_scope": sorted(section_scope),
            "section_scope_dropped": section_dropped,
            "owner_visibility": visibility.as_dict(),
            "per_owner_retrieved": per_owner,
            "per_variant_retrieved": per_variant,
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
        strategy_tags=("search", "faiss", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.search.scoped.faiss",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "file_id": {"type": "string", "description": "Restrict results to a specific file_id (required unless file_ids provided)."},
                "file_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to file_ids (required unless file_id provided)."},
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
        extra = request.extra or {}
        file_scope, invalid, raw_ids = resolve_explicit_file_scope(extra)
        if invalid:
            return ToolResult(
                summary="search.scoped.faiss skipped: invalid file_id(s) (expected UUID; use search.file to obtain file_id).",
                diagnostics={
                    "reason": "invalid_file_id_format",
                    "query": query,
                    "invalid_file_ids": invalid[:20],
                    "file_ids_filter_raw": raw_ids[:20],
                },
            )
        if file_scope is None or not getattr(file_scope, "file_ids", None):
            return ToolResult(
                summary="search.scoped.faiss skipped: missing file_id/file_ids (call search.file first).",
                diagnostics={"reason": "missing_file_scope", "query": query},
            )
        top_k = self._resolve_top_k(extra.get("top_k"), tool_defaults.SEARCH_DEFAULT_TOP_K)
        result = await self._run_faiss(request=request, query=query, top_k=top_k, file_scope=file_scope)
        return ToolResult(summary=result.summary, evidences=result.evidences, diagnostics=result.diagnostics)


class SearchGlobalFaissTool(_SearchToolBase, _FaissChannel, GraphTool):
    """FAISS-only global search tool (explicitly ignores inherited file_scope)."""

    descriptor = ToolDescriptor(
        name="search.global.faiss",
        channel="graph",
        description=(
            "FAISS-only dense search across all accessible documents. "
            "Warning: may introduce cross-document/company noise. Prefer search.scoped.faiss."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "faiss", "global", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
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
        extra = dict(request.extra or {})
        for key in ("file_id", "file_ids", "source_file_id", "source_file_ids", "filename_contains"):
            extra.pop(key, None)
        file_scope = None
        top_k = self._resolve_top_k(extra.get("top_k"), tool_defaults.SEARCH_DEFAULT_TOP_K)
        patched = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=extra,
            graph_context=patched_ctx,
            coverage_metrics=request.coverage_metrics,
        )
        result = await self._run_faiss(request=patched, query=query, top_k=top_k, file_scope=file_scope)
        risk = "global_search_may_introduce_cross_doc_noise"
        diagnostics = {**dict(result.diagnostics or {}), "search_mode": "global", "risk": risk}
        summary = (result.summary or "FAISS search completed.").rstrip()
        summary = summary + f" NOTE: {risk}."
        return ToolResult(summary=summary, evidences=result.evidences, diagnostics=diagnostics)
