"""FAISS search channel and tool."""
import asyncio
from typing import Any, Dict, List

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from framework.thread_pool import get_thread_pool

from ...base import ToolRunRequest
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
