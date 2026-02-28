"""Regex search channel for LocateTool."""

import asyncio
import logging
import re
from typing import Any, Dict, List, Sequence

from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from framework.thread_pool import get_thread_pool

from .base import _ChannelResult, _SearchToolBase

logger = logging.getLogger(__name__)


def _extract_file_ids(file_scope) -> list[str]:
    """Extract file_ids from a FileScope object (best-effort)."""
    if file_scope is None:
        return []
    raw = getattr(file_scope, "file_ids", None)
    if not raw:
        return []
    return [str(fid) for fid in raw if str(fid or "").strip()]


class _RegexChannel:
    """Regex channel: full-text regex when file_scope is available, BM25 fallback otherwise."""

    async def _run_regex(
        self: _SearchToolBase,
        *,
        request: "ToolRunRequest",
        query: str,
        top_k: int,
        file_scope,
        regex_patterns: Sequence[str],
    ) -> _ChannelResult:
        """Run regex channel: prefer full-text scan via MinerU, fallback to BM25 pool."""

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

        # Full-text path: when file_scope has file_ids, load all pages from MinerU content_list.
        file_ids = _extract_file_ids(file_scope)
        if file_ids:
            result = await self._run_regex_fulltext(
                request=request,
                query=query,
                top_k=top_k,
                file_ids=file_ids,
                compiled=compiled,
                patterns=patterns,
                invalid_patterns=invalid_patterns,
            )
            # If fulltext path returned results, use them; otherwise fall through to BM25.
            if result.evidences:
                return result

        # BM25 fallback path.
        return await self._run_regex_bm25_fallback(
            request=request,
            query=query,
            top_k=top_k,
            file_scope=file_scope,
            compiled=compiled,
            patterns=patterns,
            invalid_patterns=invalid_patterns,
        )

    async def _run_regex_fulltext(
        self: _SearchToolBase,
        *,
        request: "ToolRunRequest",
        query: str,
        top_k: int,
        file_ids: list[str],
        compiled: list[re.Pattern],
        patterns: list[str],
        invalid_patterns: list[dict],
    ) -> _ChannelResult:
        """Full-text regex: load all pages from MinerU content_list and scan."""

        from core.deepsearch.tools.explore.read_structured import _load_mineru_content_list_pages

        scored: list[tuple[str, int, str, int]] = []  # (file_id, page_idx, text, match_count)
        fulltext_diag: Dict[str, Any] = {"file_ids": file_ids, "pages_scanned": 0}

        for fid in file_ids:
            try:
                page_texts, diag = await get_thread_pool().run_blocking(
                    _load_mineru_content_list_pages,
                    file_id=fid,
                    page_start=0,
                    page_end=999999,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Regex fulltext: failed to load content_list for %s: %s", fid, exc)
                fulltext_diag.setdefault("errors", []).append({"file_id": fid, "error": str(exc)})
                continue
            if not page_texts:
                fulltext_diag.setdefault("empty_files", []).append(fid)
                continue
            fulltext_diag["pages_scanned"] = fulltext_diag.get("pages_scanned", 0) + len(page_texts)
            for page_idx, text in page_texts.items():
                total_matches = 0
                for regex in compiled:
                    total_matches += len(regex.findall(text))
                if total_matches > 0:
                    scored.append((fid, page_idx, text, total_matches))

        scored.sort(key=lambda x: (-x[3], x[1]))

        evidences: List[EvidenceChunk] = []
        for idx, (fid, page_idx, text, match_count) in enumerate(scored[:top_k]):
            snippet = self._summary_window(text)
            chunk_id = f"regex:{fid}:p{page_idx}:{idx}"
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source="regex",
                content=snippet,
                kind=EVIDENCE_KIND_DERIVED,
                score=float(match_count),
                provenance={
                    "channel": "regex",
                    "rank": idx,
                    "source_file_id": fid,
                    "page_start": page_idx,
                    "page_end": page_idx,
                    "match_count": match_count,
                    "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
                    "metadata": {"source_file_id": fid, "page_start": page_idx, "page_end": page_idx},
                },
            )
            evidences.append(evidence)

        summary = (
            f"Regex fulltext search returned {len(evidences)} page hits."
            if evidences
            else "Regex fulltext search returned no hits."
        )
        diagnostics: dict[str, Any] = {
            "query": query,
            "mode": "fulltext",
            "patterns": patterns,
            "compiled_count": len(compiled),
            "invalid_patterns": invalid_patterns,
            "total_matched_pages": len(scored),
            "fulltext": fulltext_diag,
            "results": [{"chunk_id": e.chunk_id, "score": e.score} for e in evidences],
        }
        return _ChannelResult(channel="regex", evidences=evidences, diagnostics=diagnostics, summary=summary)

    async def _run_regex_bm25_fallback(
        self: _SearchToolBase,
        *,
        request: "ToolRunRequest",
        query: str,
        top_k: int,
        file_scope,
        compiled: list[re.Pattern],
        patterns: list[str],
        invalid_patterns: list[dict],
    ) -> _ChannelResult:
        """BM25 fallback: get broad BM25 candidates, then re-score by regex match count."""

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

        pool_k = max(top_k * 2, 500)
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
            "mode": "bm25_fallback",
            "patterns": patterns,
            "compiled_count": len(compiled),
            "invalid_patterns": invalid_patterns,
            "pool_size": len(chunks),
            "pool_k": pool_k,
            "matched": len(scored),
            "file_scope_dropped": dropped,
            "results": [{"chunk_id": e.chunk_id, "score": e.score} for e in evidences],
        }
        return _ChannelResult(channel="regex", evidences=evidences, diagnostics=diagnostics, summary=summary)
