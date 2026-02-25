"""BM25 search channel and tool."""
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from framework.thread_pool import get_thread_pool

from ...base import ToolRunRequest
from .base import _ChannelResult, _SearchToolBase


class _Bm25Channel:
    @staticmethod
    def _tokenize_query(query: str, *, min_len: int) -> List[str]:
        # Tokenization for snippet highlighting only (not retrieval). Avoid regex-based tokenization.
        raw = str(query or "")
        tokens: List[str] = []
        buf: List[str] = []
        mode: Optional[str] = None  # "zh" | "alnum"

        def _flush() -> None:
            nonlocal buf, mode
            if not buf:
                return
            tokens.append("".join(buf))
            buf = []
            mode = None

        for ch in raw:
            code = ord(ch)
            is_zh = 0x4E00 <= code <= 0x9FFF
            is_alnum = ch.isalnum() or ch in {"_", "-"}
            if is_zh:
                if mode != "zh":
                    _flush()
                    mode = "zh"
                buf.append(ch)
                continue
            if is_alnum:
                if mode != "alnum":
                    _flush()
                    mode = "alnum"
                buf.append(ch)
                continue
            _flush()
        _flush()

        filtered: List[str] = []
        seen: set[str] = set()
        for token in tokens:
            key = token.strip()
            if len(key) < min_len:
                continue
            lowered = key.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            filtered.append(key)
        return filtered

    def _highlight_snippet(self: _SearchToolBase, text: str, *, tokens: Sequence[str]) -> Tuple[str, Optional[str]]:
        raw = str(text or "")
        if not raw:
            return "", None

        min_len = int(tool_defaults.SEARCH_BM25_MIN_TOKEN_LENGTH)
        query_tokens = [t for t in tokens if len(t) >= min_len]
        if not query_tokens:
            return self._summary_window(raw), None

        lowered = raw.lower()
        best_pos = None
        best_token = None
        for token in query_tokens:
            pos = lowered.find(token.lower())
            if pos == -1:
                continue
            if best_pos is None or pos < best_pos:
                best_pos = pos
                best_token = token

        if best_pos is None or best_token is None:
            return self._summary_window(raw), None

        window = max(1, int(tool_defaults.SEARCH_BM25_HIGHLIGHT_WINDOW_CHARS))
        start = max(0, int(best_pos) - int(window / 2))
        end = min(len(raw), start + window)
        snippet = raw[start:end]

        prefix = str(tool_defaults.SEARCH_BM25_HIGHLIGHT_PREFIX)
        suffix = str(tool_defaults.SEARCH_BM25_HIGHLIGHT_SUFFIX)
        # Highlight in a simple, deterministic way without regex.
        snippet_lower = snippet.lower()
        token_lower = best_token.lower()
        idx = snippet_lower.find(token_lower)
        if idx != -1:
            snippet = snippet[:idx] + prefix + snippet[idx : idx + len(best_token)] + suffix + snippet[idx + len(best_token) :]

        if start > 0:
            snippet = "..." + snippet
        if end < len(raw):
            snippet = snippet + "..."

        max_chars = max(1, int(tool_defaults.SEARCH_BM25_SNIPPET_MAX_CHARS))
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3].rstrip() + "..."
        return snippet, best_token

    async def _run_bm25(
        self: _SearchToolBase,
        *,
        request: ToolRunRequest,
        query: str,
        top_k: int,
        file_scope,
    ) -> _ChannelResult:
        retrievers = self._resolve_retrievers()
        if retrievers.bm25 is None:
            return _ChannelResult(
                channel="bm25",
                evidences=[],
                diagnostics={"query": query, "reason": "bm25_retriever_unavailable"},
                summary="BM25 search skipped: bm25 retriever unavailable.",
            )

        visibility = self._resolve_owner_visibility(request)
        if not visibility.enabled:
            return _ChannelResult(
                channel="bm25",
                evidences=[],
                diagnostics={"query": query, "reason": "owner_id_missing", "owner_visibility": visibility.as_dict()},
                summary="BM25 search skipped: missing owner scope.",
            )

        override = request.extra.get("bm25_top_k")
        effective_top_k = self._resolve_top_k(override, top_k)
        use_phrase_query = request.extra.get("bm25_use_phrase_query")
        use_phrase_query = self._coerce_bool(use_phrase_query, default=False)

        section_scope = self._resolve_section_scope(request.extra or {})

        bm25_filters = None
        if file_scope is not None and getattr(file_scope, "file_ids", None):
            bm25_filters = {"source_file_id": sorted(getattr(file_scope, "file_ids"))}
        if section_scope:
            bm25_filters = dict(bm25_filters or {})
            bm25_filters["section_id"] = sorted(section_scope)

        async def _call_one(owner_id: str, query_text: str) -> List[Chunk]:
            return await get_thread_pool().run_blocking(
                retrievers.bm25.invoke,
                query_text,
                k=effective_top_k,
                owner_id=owner_id,
                with_score=True,
                use_phrase_query=use_phrase_query,
                filters=bm25_filters,
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
            chunk_id = self._chunk_id(chunk, "bm25")
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
        chunks, section_dropped = self._apply_section_scope(chunks, section_scope)

        tokens = self._tokenize_query(query, min_len=int(tool_defaults.SEARCH_BM25_MIN_TOKEN_LENGTH))
        evidences: List[EvidenceChunk] = []
        results: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[:effective_top_k]):
            content = self._chunk_content(chunk)
            meta = self._chunk_meta(chunk)
            snippet, matched = self._highlight_snippet(content, tokens=tokens)
            chunk_id = self._chunk_id(chunk, "bm25")
            score = self._chunk_score(chunk)
            file_name = self._chunk_file_name(meta)
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source="bm25",
                content=snippet,
                kind=EVIDENCE_KIND_DERIVED,
                score=score,
                provenance={
                    "channel": "bm25",
                    "rank": idx,
                    "file_name": file_name,
                    "matched_token": matched,
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
                    "matched_token": matched,
                }
            )

        summary = f"BM25 search returned {len(evidences)} chunks." if evidences else "BM25 search returned no chunks."
        diagnostics = {
            "query": query,
            "query_variants": query_variants,
            "top_k": effective_top_k,
            "retrieved": len(chunks),
            "file_scope_dropped": dropped,
            "section_scope": sorted(section_scope),
            "section_scope_dropped": section_dropped,
            "filters_used": bm25_filters,
            "owner_visibility": visibility.as_dict(),
            "per_owner_retrieved": per_owner,
            "per_variant_retrieved": per_variant,
            "errors": errors,
            "results": results,
        }
        return _ChannelResult(channel="bm25", evidences=evidences, diagnostics=diagnostics, summary=summary)

