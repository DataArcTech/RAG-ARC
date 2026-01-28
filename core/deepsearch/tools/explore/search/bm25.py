"""BM25 search channel and tool."""
import asyncio
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from core.deepsearch.utils.file_scope import resolve_file_scope
from framework.thread_pool import get_thread_pool

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_PRIMARY, SCOPE_FILE, SCOPE_OWNER
from .base import _ChannelResult, _SearchToolBase


class _Bm25Channel:
    @staticmethod
    def _tokenize_query(query: str, *, min_len: int) -> List[str]:
        raw = str(query or "")
        tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_\\-]+", raw)
        filtered = []
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

        try:
            pattern = re.compile("|".join(re.escape(t) for t in query_tokens), flags=re.IGNORECASE)
        except re.error:
            return self._summary_window(raw), None

        match = pattern.search(raw)
        if not match:
            return self._summary_window(raw), None

        window = max(1, int(tool_defaults.SEARCH_BM25_HIGHLIGHT_WINDOW_CHARS))
        start = max(0, match.start() - int(window / 2))
        end = min(len(raw), start + window)
        snippet = raw[start:end]

        prefix = str(tool_defaults.SEARCH_BM25_HIGHLIGHT_PREFIX)
        suffix = str(tool_defaults.SEARCH_BM25_HIGHLIGHT_SUFFIX)

        def _wrap(m: re.Match[str]) -> str:
            return f"{prefix}{m.group(0)}{suffix}"

        snippet = pattern.sub(_wrap, snippet)
        if start > 0:
            snippet = "..." + snippet
        if end < len(raw):
            snippet = snippet + "..."

        max_chars = max(1, int(tool_defaults.SEARCH_BM25_SNIPPET_MAX_CHARS))
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3].rstrip() + "..."
        return snippet, match.group(0)

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

        async def _call_one(owner_id: str) -> List[Chunk]:
            return await get_thread_pool().run_blocking(
                retrievers.bm25.invoke,
                query,
                k=effective_top_k,
                owner_id=owner_id,
                with_score=True,
                use_phrase_query=use_phrase_query,
                filters=bm25_filters,
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
                kind=EVIDENCE_KIND_PRIMARY,
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
            "top_k": effective_top_k,
            "retrieved": len(chunks),
            "file_scope_dropped": dropped,
            "section_scope": sorted(section_scope),
            "section_scope_dropped": section_dropped,
            "filters_used": bm25_filters,
            "owner_visibility": visibility.as_dict(),
            "per_owner_retrieved": per_owner,
            "errors": errors,
            "results": results,
        }
        return _ChannelResult(channel="bm25", evidences=evidences, diagnostics=diagnostics, summary=summary)


class SearchBM25Tool(_SearchToolBase, _Bm25Channel, GraphTool):
    """BM25-only search tool."""

    descriptor = ToolDescriptor(
        name="search.bm25",
        channel="graph",
        description="BM25-only lexical search with highlighted snippets.",
        speed="fast",
        cost="low",
        strategy_tags=("search", "bm25", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.search.bm25",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "file_id": {"type": "string", "description": "Restrict results to a specific file_id."},
                "file_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to file_ids."},
                "filename_contains": {"type": "array", "items": {"type": "string"}, "description": "Best-effort filename filter."},
                "section_id": {"type": "string", "description": "Restrict results to a specific section_id."},
                "section_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to section_ids."},
                "owner_ids": {"type": "array", "items": {"type": "string"}, "description": "Owner ids to search (me/share)."},
                "top_k": {"type": "integer", "minimum": 0, "description": "Top-k results to return."},
                "bm25_top_k": {"type": "integer", "minimum": 0, "description": "Alias of top_k."},
                "bm25_use_phrase_query": {"type": "boolean", "description": "Use phrase query for BM25."},
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
        result = await self._run_bm25(request=request, query=query, top_k=top_k, file_scope=file_scope)
        return ToolResult(summary=result.summary, evidences=result.evidences, diagnostics=result.diagnostics)
