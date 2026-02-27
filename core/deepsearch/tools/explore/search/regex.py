"""Regex search channel for LocateTool."""

import asyncio
import re
from typing import Any, List, Sequence

from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from framework.thread_pool import get_thread_pool

from .base import _ChannelResult, _SearchToolBase


class _RegexChannel:
    """Regex channel: re-score BM25 candidate chunks by regex match count."""

    async def _run_regex(
        self: _SearchToolBase,
        *,
        request: "ToolRunRequest",
        query: str,
        top_k: int,
        file_scope,
        regex_patterns: Sequence[str],
    ) -> _ChannelResult:
        """Run regex channel over BM25 candidate pool.

        Strategy: get broad BM25 candidates, then re-score by regex match count.
        Invalid patterns are silently skipped.
        """

        patterns = [p for p in regex_patterns if isinstance(p, str) and p.strip()]
        if not patterns:
            return _ChannelResult(
                channel="regex",
                evidences=[],
                diagnostics={"query": query, "reason": "no_patterns"},
                summary="Regex search skipped: no patterns.",
            )

        compiled = []
        invalid_patterns = []
        for p in patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE | re.MULTILINE))
            except re.error as exc:
                invalid_patterns.append({"pattern": p, "error": str(exc)})

        if not compiled:
            return _ChannelResult(
                channel="regex",
                evidences=[],
                diagnostics={"query": query, "reason": "all_patterns_invalid", "invalid": invalid_patterns},
                summary="Regex search skipped: all patterns invalid.",
            )

        retrievers = self._resolve_retrievers()
        if retrievers.bm25 is None:
            return _ChannelResult(
                channel="regex",
                evidences=[],
                diagnostics={"query": query, "reason": "bm25_retriever_unavailable"},
                summary="Regex search skipped: bm25 retriever unavailable (needed for candidate pool).",
            )

        visibility = self._resolve_owner_visibility(request)
        if not visibility.enabled:
            return _ChannelResult(
                channel="regex",
                evidences=[],
                diagnostics={"query": query, "reason": "owner_id_missing"},
                summary="Regex search skipped: missing owner scope.",
            )

        pool_k = max(top_k * 2, 50)
        owner_ids = list(visibility.owner_ids_used)

        async def _call_bm25(owner_id: str) -> List[Chunk]:
            return await get_thread_pool().run_blocking(
                retrievers.bm25.invoke,
                query,
                k=pool_k,
                owner_id=owner_id,
                with_score=True,
            )

        parts = await asyncio.gather(*[_call_bm25(oid) for oid in owner_ids], return_exceptions=True)
        chunks: List[Chunk] = []
        for part in parts:
            if isinstance(part, Exception):
                continue
            chunks.extend(part)

        chunks, dropped = self._apply_file_scope(chunks, file_scope)

        scored: List[tuple[Chunk, int]] = []
        for chunk in chunks:
            content = self._chunk_content(chunk)
            if not content:
                continue
            total_matches = 0
            for regex in compiled:
                total_matches += len(regex.findall(content))
            if total_matches > 0:
                scored.append((chunk, total_matches))

        scored.sort(key=lambda x: x[1], reverse=True)

        evidences: List[EvidenceChunk] = []
        for idx, (chunk, match_count) in enumerate(scored[:top_k]):
            content = self._chunk_content(chunk)
            meta = self._chunk_meta(chunk)
            snippet = self._summary_window(content)
            chunk_id = self._chunk_id(chunk, "regex")
            score = float(match_count)
            file_name = self._chunk_file_name(meta)
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source="regex",
                content=snippet,
                kind=EVIDENCE_KIND_DERIVED,
                score=score,
                provenance={
                    "channel": "regex",
                    "rank": idx,
                    "file_name": file_name,
                    "match_count": match_count,
                    "owner_id": getattr(chunk, "owner_id", None) or meta.get("owner_id"),
                    "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
                    "metadata": meta,
                },
            )
            evidences.append(evidence)

        summary = f"Regex search returned {len(evidences)} chunks." if evidences else "Regex search returned no chunks."
        diagnostics: dict[str, Any] = {
            "query": query,
            "patterns": patterns,
            "compiled_count": len(compiled),
            "invalid_patterns": invalid_patterns,
            "pool_size": len(chunks),
            "matched": len(scored),
            "file_scope_dropped": dropped,
            "results": [{"chunk_id": e.chunk_id, "score": e.score} for e in evidences],
        }
        return _ChannelResult(channel="regex", evidences=evidences, diagnostics=diagnostics, summary=summary)
