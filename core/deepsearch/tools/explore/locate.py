"""Unified locate tool: file-level routing via multi-channel retrieval."""
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from config.core.deepsearch import tool_defaults
from core.deepsearch.utils.file_scope import FileScope
from core.deepsearch.utils.ids import coerce_uuid_list, resolve_file_ref
from core.deepsearch.utils.llm_envelope import build_llm_envelope

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from .search.base import _ChannelResult, _SearchToolBase
from .search.bm25 import _Bm25Channel
from .search.faiss import _FaissChannel
from .search.graph_chunk import _GraphChunkChannel
from .search.regex import _RegexChannel
from .search.structure import _StructureChannel
from core.deepsearch.utils.section_tree import SectionNode, fetch_section_nodes

logger = logging.getLogger(__name__)


def _coerce_file_id(meta: Mapping[str, Any]) -> Optional[str]:
    for key in ("source_file_id", "file_id", "doc_id", "document_id"):
        token = str(meta.get(key) or "").strip()
        if token:
            return token
    return None


def _coerce_filename(meta: Mapping[str, Any]) -> Optional[str]:
    for key in ("filename", "source_file_name", "file_name", "source_path", "path"):
        token = str(meta.get(key) or "").strip()
        if token:
            return token
    return None


def _macro_filename(value: Optional[str]) -> Optional[str]:
    token = str(value or "").strip()
    if not token:
        return None
    normalized = token.replace("\\", "/")
    return os.path.basename(normalized) or token


def _rrf(rank: int, *, k: int) -> float:
    return 1.0 / float(max(1, int(k) + int(rank)))


def _extract_page(meta: Mapping[str, Any]) -> Optional[int]:
    """Extract page number from chunk metadata (multiple key conventions)."""
    for key in ("page_start", "page_idx", "page_num"):
        val = meta.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return None


@dataclass
class _Hit:
    channel: str
    chunk_id: str
    rank: int
    score: Optional[float]
    snippet: str
    page: Optional[int] = None


class LocateTool(_SearchToolBase, _FaissChannel, _Bm25Channel, _GraphChunkChannel, _RegexChannel, _StructureChannel, GraphTool):
    """Unified locate tool: find relevant files via multi-channel retrieval + RRF fusion."""

    @staticmethod
    def _query_has_rerank_skip_block_cues(query: str) -> bool:
        text = str(query or "")
        lowered = text.lower()
        cues = getattr(tool_defaults, "FILE_SEARCH_RERANK_SKIP_BLOCK_QUERY_CUES", ()) or ()
        for cue in cues:
            token = str(cue or "")
            if not token:
                continue
            if token.lower() in lowered:
                return True
            if token in text:
                return True
        return False

    @classmethod
    def _should_skip_rerank(cls, *, query: str, candidates: Sequence[Mapping[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        """Return (skip, diagnostics) for the rerank API call."""

        diag: Dict[str, Any] = {"policy": "score_margin_ratio", "skip": False}
        items = list(candidates or [])
        if len(items) < 2:
            diag["reason"] = "too_few_candidates"
            return False, diag

        try:
            top1 = float(items[0].get("score") or 0.0)
            top2 = float(items[1].get("score") or 0.0)
        except Exception:
            diag["reason"] = "invalid_scores"
            return False, diag

        threshold = float(getattr(tool_defaults, "FILE_SEARCH_RERANK_SKIP_SCORE_MARGIN_RATIO", 0.0) or 0.0)
        threshold = max(0.0, threshold)
        diag["threshold_ratio"] = threshold
        diag["top1_score"] = top1
        diag["top2_score"] = top2

        if threshold <= 0:
            diag["reason"] = "threshold_disabled"
            return False, diag

        if cls._query_has_rerank_skip_block_cues(query):
            diag["reason"] = "blocked_by_query_cue"
            return False, diag

        denom = max(abs(top1), 1e-9)
        margin_ratio = (top1 - top2) / denom
        diag["margin_ratio"] = margin_ratio
        if margin_ratio > threshold:
            diag["skip"] = True
            diag["reason"] = "confident_top1"
            return True, diag
        diag["reason"] = "margin_too_small"
        return False, diag

    descriptor = ToolDescriptor(
        name="locate",
        channel="graph",
        description=(
            "Search for relevant files or pages. "
            "Without file_id: returns a ranked list of files across the knowledge base (file-level routing). "
            "With file_id: returns ranked pages within that single file (page-level routing). "
            "Results are navigation snippets only — NOT citeable evidence; always follow up with read.pages."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "locate", "file_search", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.locate",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {
                    "type": "string",
                    "description": (
                        "A refined search query to use instead of the main question. "
                        "Use this to narrow the search with specific keywords or phrases "
                        "(e.g. 'consolidated balance sheet FY2024'). "
                        "When omitted, the main question is used as the search query."
                    ),
                },
                "file_id": {
                    "type": "string",
                    "description": (
                        "A file_id (UUID) or filename to search within. When set, switches to page-level mode "
                        "and returns ranked pages instead of files. "
                        "Omit this to search across all files."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Maximum number of results to return. "
                        "Returns top_k files (file-level) or top_k pages (page-level). Defaults to 5."
                    ),
                },
                "regex_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Regex patterns for exact keyword matching. Use when you know specific terms "
                        "that must appear (e.g. ['appendix\\\\s+[A-Z]', 'revised.*2018', 'FY\\\\s*202[0-9]']). "
                        "Patterns are case-insensitive. Complements the semantic search channels."
                    ),
                },
            }
        ),
        example_args={
            "question": "Which PDF mentions multi-currency switching?",
            "plan_step": "plan_01",
            "extra": {"top_k": 5},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)

        query_spec: Dict[str, Any] = {}
        if request.graph_context and isinstance(request.graph_context.metadata, dict):
            raw = request.graph_context.metadata.get("query_spec")
            if isinstance(raw, dict):
                query_spec = dict(raw)

        raw_bm25_terms = query_spec.get("bm25_terms") or []
        if not isinstance(raw_bm25_terms, (list, tuple)):
            raw_bm25_terms = []
        bm25_extra_terms = [str(x).strip() for x in raw_bm25_terms if str(x).strip()]

        raw_regex = query_spec.get("regex_patterns") or []
        if not isinstance(raw_regex, (list, tuple)):
            raw_regex = []
        regex_patterns = [str(x).strip() for x in raw_regex if str(x).strip()]

        # Agent-provided regex_patterns (via tool_args) merged with QuerySpec patterns.
        agent_regex = (request.extra or {}).get("regex_patterns") or []
        if isinstance(agent_regex, (list, tuple)):
            agent_patterns = [str(x).strip() for x in agent_regex if str(x).strip()]
            if agent_patterns:
                seen_pat = set(agent_patterns)
                merged = list(agent_patterns)
                for p in regex_patterns:
                    if p not in seen_pat:
                        merged.append(p)
                        seen_pat.add(p)
                regex_patterns = merged

        visibility = self._resolve_owner_visibility(request)
        diagnostics: Dict[str, Any] = {
            "query": query,
            "bm25_extra_terms": bm25_extra_terms,
            "regex_patterns": regex_patterns,
            "owner_visibility": visibility.as_dict(),
            "query_spec_present": bool(query_spec),
        }

        if not visibility.enabled:
            return ToolResult(
                summary="locate skipped: missing owner scope.",
                diagnostics={**diagnostics, "reason": "owner_scope_missing"},
            )

        file_scope = None
        resolved_file_id, file_ref_raw = await resolve_file_ref(
            request.extra or {}, adapter=request.adapter, access_scope=request.access_scope,
        )
        if resolved_file_id:
            file_scope = FileScope(file_ids=frozenset([resolved_file_id]), filename_contains=(), source="locate_arg")
            diagnostics["file_scope"] = {"file_id": resolved_file_id, "raw": file_ref_raw}
        elif file_ref_raw:
            diagnostics["file_scope"] = {"file_id": None, "raw": file_ref_raw, "reason": "unresolved"}

        # Page-level mode: when file= is specified, switch to page aggregation.
        if file_scope is not None:
            return await self._run_page_level(
                request=request, query=query,
                bm25_extra_terms=bm25_extra_terms, regex_patterns=regex_patterns,
                file_scope=file_scope, diagnostics=diagnostics,
            )

        top_k_files = self._resolve_top_k((request.extra or {}).get("top_k"), tool_defaults.FILE_SEARCH_DEFAULT_TOP_K)
        top_k_chunks = int(tool_defaults.FILE_SEARCH_CHANNEL_TOP_K)

        async def _run_faiss_seeded() -> _ChannelResult:
            """Run FAISS for the query."""
            extra = dict(request.extra or {})
            extra["focus_query"] = query
            req = ToolRunRequest(
                question=request.question, plan_step=request.plan_step,
                context_evidences=request.context_evidences,
                adapter=request.adapter, access_scope=request.access_scope,
                extra=extra, graph_context=request.graph_context,
                coverage_metrics=request.coverage_metrics,
            )
            return await self._run_faiss(request=req, query=query, top_k=top_k_chunks, file_scope=file_scope)

        async def _run_bm25_seeded() -> _ChannelResult:
            variants = [query] + [t for t in bm25_extra_terms if t != query]
            parts = await asyncio.gather(
                *[
                    self._run_bm25(request=request, query=qv, top_k=top_k_chunks, file_scope=file_scope)
                    for qv in variants
                ],
                return_exceptions=True,
            )
            merged_evidences: List[Any] = []
            merged_diags: Dict[str, Any] = {"variants": []}
            summaries: List[str] = []
            for qv, part in zip(variants, parts):
                if isinstance(part, Exception):
                    merged_diags["variants"].append({"query": qv, "error": str(part)})
                    continue
                summaries.append(part.summary)
                merged_evidences.extend(part.evidences or [])
                merged_diags["variants"].append({"query": qv, "diagnostics": dict(part.diagnostics or {})})
            summary = " ".join([s for s in summaries if s]).strip() or "BM25 search completed."
            return _ChannelResult(
                channel="bm25",
                evidences=merged_evidences,
                diagnostics={"query": query, "seeded_variants": variants, **merged_diags},
                summary=summary,
            )

        tasks: List[tuple[str, asyncio.Task]] = []
        tasks.append(("faiss", asyncio.create_task(_run_faiss_seeded())))
        tasks.append(("bm25", asyncio.create_task(_run_bm25_seeded())))
        tasks.append(
            (
                "graph_chunk",
                asyncio.create_task(
                    self._run_graph_chunk(request=request, query=query, top_k=top_k_chunks, file_scope=file_scope)
                ),
            )
        )
        if regex_patterns:
            tasks.append(
                (
                    "regex",
                    asyncio.create_task(
                        self._run_regex(
                            request=request,
                            query=query,
                            top_k=top_k_chunks,
                            file_scope=file_scope,
                            regex_patterns=regex_patterns,
                        )
                    ),
                )
            )

        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        channel_summaries: List[str] = []
        channel_diags: Dict[str, Any] = {}
        channel_results: Dict[str, Any] = {}
        for (channel, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                channel_summaries.append(f"{channel} search failed.")
                channel_diags[channel] = {"query": query, "error": str(result)}
                continue
            channel_results[channel] = result
            channel_summaries.append(result.summary)
            channel_diags[channel] = dict(result.diagnostics or {})

        diagnostics["channels_summary"] = channel_summaries
        diagnostics["channels_diagnostics"] = channel_diags

        hits: Dict[str, Dict[str, Any]] = {}
        seen_chunk_ids_by_file: Dict[str, set[str]] = {}
        per_channel_scores: Dict[str, Dict[str, List[float]]] = {}
        rrf_k = int(tool_defaults.FILE_SEARCH_RRF_K)

        # File-level dedup: merge different file_ids that share the same
        # physical filename so re-indexed or stale-index chunks aggregate
        # into a single result row instead of appearing as duplicates.
        _filename_to_canonical_fid: Dict[str, str] = {}

        def _canonical_file_id(raw_fid: str, meta: Mapping[str, Any]) -> str:
            """Return the canonical file_id for a given raw file_id + metadata."""
            macro = _macro_filename(_coerce_filename(meta))
            if not macro:
                return raw_fid
            existing = _filename_to_canonical_fid.get(macro)
            if existing is not None:
                return existing
            _filename_to_canonical_fid[macro] = raw_fid
            return raw_fid

        async def _consume(channel: str, evidences: Sequence[Any], *, diag: Mapping[str, Any]) -> None:
            channel_diags[channel] = dict(diag or {})
            for ev in evidences or []:
                provenance = getattr(ev, "provenance", None) or {}
                meta = provenance.get("metadata") if isinstance(provenance, dict) else None
                meta = dict(meta or {}) if isinstance(meta, dict) else {}
                raw_fid = _coerce_file_id(meta)
                if not raw_fid:
                    continue
                file_id = _canonical_file_id(raw_fid, meta)

                chunk_id = str(getattr(ev, "chunk_id", "") or "").strip()
                if not chunk_id:
                    continue
                seen = seen_chunk_ids_by_file.setdefault(file_id, set())
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)

                rank = provenance.get("rank") if isinstance(provenance, dict) else None
                try:
                    rank_i = int(rank) + 1 if rank is not None else 9999
                except Exception:
                    rank_i = 9999

                score_val = getattr(ev, "score", None)
                score_f = float(score_val) if isinstance(score_val, (int, float)) else None
                snippet = str(getattr(ev, "content", "") or "").strip()

                row = hits.get(file_id)
                if row is None:
                    macro_name = _macro_filename(_coerce_filename(meta))
                    row = {
                        "file_id": file_id,
                        "filename": macro_name,
                        "score": 0.0,
                        "hit_count": 0,
                        "hits": [],
                        "per_channel_hits": {},
                    }
                    hits[file_id] = row

                row["hit_count"] = int(row.get("hit_count") or 0) + 1
                per_channel = dict(row.get("per_channel_hits") or {})
                per_channel[channel] = int(per_channel.get(channel) or 0) + 1
                row["per_channel_hits"] = per_channel
                if score_f is not None:
                    channel_scores = per_channel_scores.setdefault(channel, {})
                    channel_scores.setdefault(file_id, []).append(score_f)
                # Extract page from provenance (regex/structure) or metadata (faiss/bm25/graph).
                page: Optional[int] = None
                if isinstance(provenance, dict):
                    for _pk in ("page_start", "page_idx"):
                        _pv = provenance.get(_pk)
                        if _pv is not None:
                            try:
                                page = int(_pv)
                            except (ValueError, TypeError):
                                pass
                            else:
                                break
                if page is None:
                    page = _extract_page(meta)

                row["hits"].append(
                    _Hit(
                        channel=channel,
                        chunk_id=chunk_id,
                        rank=rank_i,
                        score=score_f,
                        snippet=snippet,
                        page=page,
                    ).__dict__
                )

        if "faiss" in channel_results:
            result = channel_results["faiss"]
            await _consume("faiss", result.evidences, diag=result.diagnostics)
        if "bm25" in channel_results:
            result = channel_results["bm25"]
            await _consume("bm25", result.evidences, diag=result.diagnostics)
        if "graph_chunk" in channel_results:
            result = channel_results["graph_chunk"]
            await _consume("graph_chunk", result.evidences, diag=result.diagnostics)
        if "regex" in channel_results:
            result = channel_results["regex"]
            await _consume("regex", result.evidences, diag=result.diagnostics)

        channel_doc_scores: Dict[str, Dict[str, float]] = {}
        channel_doc_ranks: Dict[str, Dict[str, int]] = {}
        fused_scores: Dict[str, float] = {}

        for channel, file_scores in per_channel_scores.items():
            doc_scores: Dict[str, float] = {}
            for file_id, scores in file_scores.items():
                n = len(scores)
                if n <= 0:
                    doc_scores[file_id] = 0.0
                    continue
                doc_scores[file_id] = float(sum(scores)) / float((n + 1) ** 0.5)
            channel_doc_scores[channel] = doc_scores

            ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], str(item[0])))
            ranks: Dict[str, int] = {}
            for idx, (file_id, _) in enumerate(ranked, start=1):
                ranks[file_id] = idx
                fused_scores[file_id] = float(fused_scores.get(file_id) or 0.0) + _rrf(idx, k=rrf_k)
            channel_doc_ranks[channel] = ranks

        for file_id, row in hits.items():
            row["score"] = float(fused_scores.get(file_id) or 0.0)
            row["per_channel_docscore"] = {ch: channel_doc_scores.get(ch, {}).get(file_id) for ch in channel_doc_scores}
            row["per_channel_docrank"] = {ch: channel_doc_ranks.get(ch, {}).get(file_id) for ch in channel_doc_ranks}

        diagnostics["channel_doc_scores"] = channel_doc_scores
        diagnostics["channel_doc_ranks"] = channel_doc_ranks

        all_results = sorted(hits.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)
        rerank_diag: Dict[str, Any] = {}
        rerank_enabled = self._coerce_bool(
            (request.extra or {}).get("enable_api_rerank"),
            tool_defaults.FILE_SEARCH_ENABLE_API_RERANK_DEFAULT,
        )
        rerank_top_k = self._resolve_top_k((request.extra or {}).get("rerank_top_k"), tool_defaults.FILE_SEARCH_RERANK_TOP_K)
        if rerank_enabled and all_results and rerank_top_k != 0:
            skip, skip_diag = self._should_skip_rerank(query=query, candidates=all_results)
            if skip:
                rerank_diag = {"enabled": False, **skip_diag}
            else:
                rerank_diag = {"enabled": True, **skip_diag}
            limited = all_results[:rerank_top_k] if rerank_top_k > 0 else all_results
            if not skip:
                reranked_ids, api_diag = await self._api_rerank(query=query, candidates=limited)
                rerank_diag.update(api_diag)
                if reranked_ids:
                    remaining = [row for row in all_results if row.get("file_id") not in reranked_ids]
                    order_map = {fid: idx for idx, fid in enumerate(reranked_ids)}
                    limited_sorted = sorted(
                        [row for row in all_results if row.get("file_id") in order_map],
                        key=lambda row: order_map.get(row.get("file_id"), 9999),
                    )
                    all_results = limited_sorted + remaining
        if rerank_diag:
            diagnostics["api_rerank"] = rerank_diag

        results_out = all_results[:top_k_files] if top_k_files > 0 else []
        diagnostics["results"] = results_out

        if not results_out:
            reason = "empty_hit"
            if visibility.owner_ids_rejected:
                reason = "owner_ids_rejected"
            return ToolResult(
                summary=build_llm_envelope(
                    thinking="File routing via multi-channel chunk retrieval produced no usable file candidates.",
                    answer=[],
                    extra={"reason": reason, "next_step": "Revise query terms; then rerun locate."},
                ),
                diagnostics={**diagnostics, "reason": reason},
            )

        max_hits = max(1, int(tool_defaults.FILE_SEARCH_MAX_SNIPPETS_PER_FILE))
        max_preview = max(0, int(tool_defaults.FILE_SEARCH_SUMMARY_SNIPPET_PREVIEW_CHARS))
        preview_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(results_out[: min(len(results_out), 5)]):
            preview = {
                "rank": idx + 1,
                "file_id": row.get("file_id"),
                "filename": row.get("filename") or "",
                "score": float(row.get("score") or 0.0),
                "hit_count": int(row.get("hit_count") or 0),
                "per_channel_hits": row.get("per_channel_hits") or {},
                "snippets": [],
            }
            hit_items = list(row.get("hits") or [])
            hit_items.sort(key=lambda item: (str(item.get("channel") or ""), int(item.get("rank") or 9999)))
            shown = 0
            for item in hit_items:
                if shown >= max_hits:
                    break
                snippet = str(item.get("snippet") or "").strip()
                if max_preview and len(snippet) > max_preview:
                    snippet = snippet[: max(0, max_preview - 3)].rstrip() + "..."
                if not snippet:
                    continue
                preview["snippets"].append(
                    {
                        "channel": str(item.get("channel") or ""),
                        "rank": int(item.get("rank") or 0),
                        "page": item.get("page"),
                        "snippet": snippet,
                    }
                )
                shown += 1
            preview_rows.append(preview)

        next_steps = (
            "Pick the best file_id from answer[0], then use toc.tree "
            "to locate relevant pages, and finally use read.pages for citeable evidence."
        )

        ranked_file_ids = [str(row.get("file_id")) for row in results_out if row.get("file_id")]
        summary = build_llm_envelope(
            thinking="Aggregate chunk hits into file candidates (routing only), optionally rerank by relevance.",
            answer=ranked_file_ids,
            extra={
                "preview": preview_rows,
                "next_steps": next_steps,
            },
        )
        return ToolResult(summary=summary, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # Page-level locate (file= is set)
    # ------------------------------------------------------------------

    async def _run_page_level(
        self,
        *,
        request: ToolRunRequest,
        query: str,
        bm25_extra_terms: List[str],
        regex_patterns: List[str],
        file_scope: FileScope,
        diagnostics: Dict[str, Any],
    ) -> ToolResult:
        """Page-level locate: 5-channel retrieval -> page aggregation -> RRF -> tree propagation -> rerank."""
        file_id = next(iter(file_scope.file_ids))
        top_k_chunks = int(getattr(tool_defaults, "PAGE_LOCATE_CHANNEL_TOP_K", tool_defaults.FILE_SEARCH_CHANNEL_TOP_K))
        top_k_pages = self._resolve_top_k(
            (request.extra or {}).get("top_k"),
            getattr(tool_defaults, "PAGE_LOCATE_DEFAULT_TOP_K", tool_defaults.FILE_SEARCH_DEFAULT_TOP_K),
        )
        rrf_k = int(tool_defaults.FILE_SEARCH_RRF_K)

        # Fetch sections once (shared by structure channel + tree propagation).
        sections: List[SectionNode] = []
        if request.adapter is not None and request.access_scope is not None:
            sections, sec_diag = await fetch_section_nodes(
                adapter=request.adapter, access_scope=request.access_scope, file_id=file_id,
            )
            diagnostics["sections"] = sec_diag

        # Build page->section lookup once (title + page range for section_context).
        page_to_sections: Dict[int, List[str]] = {}
        page_to_section_context: Dict[int, List[Dict[str, Any]]] = {}
        for sec in sections:
            if sec.page_start is None:
                continue
            p_end = sec.page_end if sec.page_end is not None else sec.page_start
            entry = {"title": sec.title or sec.path, "page_start": sec.page_start, "page_end": p_end}
            for p in range(sec.page_start, p_end + 1):
                page_to_sections.setdefault(p, []).append(sec.title or sec.path)
                page_to_section_context.setdefault(p, []).append(entry)

        async def _run_faiss_seeded() -> _ChannelResult:
            """Run FAISS for the query."""
            extra = dict(request.extra or {})
            extra["focus_query"] = query
            req = ToolRunRequest(
                question=request.question, plan_step=request.plan_step,
                context_evidences=request.context_evidences,
                adapter=request.adapter, access_scope=request.access_scope,
                extra=extra, graph_context=request.graph_context,
                coverage_metrics=request.coverage_metrics,
            )
            return await self._run_faiss(request=req, query=query, top_k=top_k_chunks, file_scope=file_scope)

        async def _run_bm25_seeded() -> _ChannelResult:
            variants = [query] + [t for t in bm25_extra_terms if t != query]
            parts = await asyncio.gather(
                *[self._run_bm25(request=request, query=qv, top_k=top_k_chunks, file_scope=file_scope) for qv in variants],
                return_exceptions=True,
            )
            merged_evs: List[Any] = []
            merged_diags: Dict[str, Any] = {"variants": []}
            summaries: List[str] = []
            for qv, part in zip(variants, parts):
                if isinstance(part, Exception):
                    merged_diags["variants"].append({"query": qv, "error": str(part)})
                    continue
                summaries.append(part.summary)
                merged_evs.extend(part.evidences or [])
                merged_diags["variants"].append({"query": qv, "diagnostics": dict(part.diagnostics or {})})
            return _ChannelResult(
                channel="bm25", evidences=merged_evs, diagnostics=merged_diags,
                summary=" ".join(s for s in summaries if s).strip() or "BM25 search completed.",
            )

        # Launch all channels concurrently.
        tasks: List[tuple[str, asyncio.Task]] = [
            ("faiss", asyncio.create_task(_run_faiss_seeded())),
            ("bm25", asyncio.create_task(_run_bm25_seeded())),
            ("graph_chunk", asyncio.create_task(
                self._run_graph_chunk(request=request, query=query, top_k=top_k_chunks, file_scope=file_scope))),
        ]
        if regex_patterns:
            tasks.append(("regex", asyncio.create_task(self._run_regex(
                request=request, query=query, top_k=top_k_chunks,
                file_scope=file_scope, regex_patterns=regex_patterns,
            ))))
        if sections:
            tasks.append(("structure", asyncio.create_task(self._run_structure(
                request=request, query=query, file_id=file_id, sections=sections,
            ))))

        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
        channel_results: Dict[str, _ChannelResult] = {}
        channel_diags: Dict[str, Any] = {}
        for (ch, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                channel_diags[ch] = {"error": str(result)}
                continue
            channel_results[ch] = result
            channel_diags[ch] = dict(result.diagnostics or {})
        diagnostics["channels_diagnostics"] = channel_diags

        # ---- Page aggregation ----
        per_channel_page_scores: Dict[str, Dict[int, List[float]]] = {}
        page_snippets: Dict[int, List[Dict[str, Any]]] = {}

        for ch_name, ch_result in channel_results.items():
            if ch_name == "structure":
                # Structure channel: credit each page in section's page range.
                bucket = per_channel_page_scores.setdefault("structure", {})
                for ev in ch_result.evidences or []:
                    prov = getattr(ev, "provenance", None) or {}
                    meta = prov.get("metadata") if isinstance(prov, dict) else {}
                    meta = dict(meta) if isinstance(meta, dict) else {}
                    p_start = meta.get("page_start")
                    if p_start is None:
                        continue
                    score_f = float(getattr(ev, "score", 0) or 0)
                    if score_f <= 0:
                        continue
                    p_end = meta.get("page_end", p_start)
                    for p in range(int(p_start), int(p_end) + 1):
                        bucket.setdefault(p, []).append(score_f)
                continue

            # Chunk channels: extract page from metadata.
            for ev in ch_result.evidences or []:
                prov = getattr(ev, "provenance", None) or {}
                meta = prov.get("metadata") if isinstance(prov, dict) else {}
                meta = dict(meta) if isinstance(meta, dict) else {}
                page = _extract_page(meta)
                if page is None:
                    continue
                score_f = getattr(ev, "score", None)
                score_f = float(score_f) if isinstance(score_f, (int, float)) else None
                if score_f is not None and score_f > 0:
                    per_channel_page_scores.setdefault(ch_name, {}).setdefault(page, []).append(score_f)
                snippet = str(getattr(ev, "content", "") or "").strip()
                if snippet:
                    rank_raw = prov.get("rank") if isinstance(prov, dict) else None
                    try:
                        rank_i = int(rank_raw) + 1 if rank_raw is not None else 9999
                    except Exception:
                        rank_i = 9999
                    page_snippets.setdefault(page, []).append({"channel": ch_name, "rank": rank_i, "snippet": snippet})

        # Per-channel doc-score per page: sum(score)/sqrt(n+1).
        channel_page_docscores: Dict[str, Dict[int, float]] = {}
        for ch, page_scores in per_channel_page_scores.items():
            docscores: Dict[int, float] = {}
            for page, scores in page_scores.items():
                n = len(scores)
                docscores[page] = sum(scores) / (n + 1) ** 0.5 if n > 0 else 0.0
            channel_page_docscores[ch] = docscores

        # RRF fusion across channels.
        fused: Dict[int, float] = {}
        for ch, docscores in channel_page_docscores.items():
            ranked = sorted(docscores.items(), key=lambda x: (-x[1], x[0]))
            for rank_idx, (page, _) in enumerate(ranked, start=1):
                fused[page] = fused.get(page, 0.0) + _rrf(rank_idx, k=rrf_k)

        # ---- Tree propagation ----
        # Boost non-hit pages in sections that contain hit pages.
        propagation_weight = float(getattr(tool_defaults, "STRUCTURE_TREE_PROPAGATION_WEIGHT", 0.25) or 0.25)
        if sections and fused and propagation_weight > 0:
            hit_pages = set(fused.keys())
            for sec in sections:
                if sec.page_start is None:
                    continue
                p_end = sec.page_end if sec.page_end is not None else sec.page_start
                sec_pages = set(range(sec.page_start, p_end + 1))
                sec_hit_pages = sec_pages & hit_pages
                if not sec_hit_pages:
                    continue
                max_hit_score = max(fused[p] for p in sec_hit_pages)
                boost = max_hit_score * propagation_weight
                for p in sec_pages - hit_pages:
                    fused[p] = fused.get(p, 0.0) + boost

        if not fused:
            return ToolResult(
                summary=build_llm_envelope(
                    thinking="Page-level locate found no page candidates within the target file.",
                    answer=[], extra={"reason": "no_page_hits", "file_id": file_id},
                ),
                diagnostics={**diagnostics, "reason": "no_page_hits"},
            )

        ranked_pages = sorted(fused.items(), key=lambda x: (-x[1], x[0]))

        # ---- API rerank (page-level: always ON) ----
        rerank_diag: Dict[str, Any] = {}
        rerank_top_k = self._resolve_top_k(
            (request.extra or {}).get("rerank_top_k"),
            getattr(tool_defaults, "PAGE_LOCATE_RERANK_TOP_K", tool_defaults.FILE_SEARCH_RERANK_TOP_K),
        )
        if rerank_top_k > 0 and len(ranked_pages) > 1:
            max_preview = max(0, int(tool_defaults.FILE_SEARCH_SUMMARY_SNIPPET_PREVIEW_CHARS))
            limited = ranked_pages[:rerank_top_k]
            rerank_docs: List[str] = []
            rerank_page_ids: List[int] = []
            for page, _score in limited:
                snips = page_snippets.get(page, [])
                snips.sort(key=lambda s: s.get("rank", 9999))
                top_snips = [s["snippet"][:max_preview] for s in snips[:3]]
                sec_titles = page_to_sections.get(page, [])
                header = f"[Page {page}]"
                if sec_titles:
                    header += f" Section: {sec_titles[0]}"
                rerank_docs.append(header + "\n" + "\n".join(top_snips) if top_snips else header)
                rerank_page_ids.append(page)

            client = self._get_rerank_client()
            if client is not None:
                _page_rerank_instruct = (
                    f"User query is: {query}\n"
                    "Select the page most likely to contain the answer to this query."
                )
                try:
                    scored = await client.arerank(
                        query=query, documents=rerank_docs, top_k=len(rerank_docs),
                        instruct=_page_rerank_instruct,
                    )
                    reranked_order = [rerank_page_ids[idx] for idx, _ in scored if 0 <= idx < len(rerank_page_ids)]
                    reranked_set = set(reranked_order)
                    remaining = [(p, s) for p, s in ranked_pages if p not in reranked_set]
                    ranked_pages = [(p, fused.get(p, 0.0)) for p in reranked_order] + remaining
                    rerank_diag = {"enabled": True, "model": getattr(client, "model_name", None), "reranked": len(reranked_order)}
                except Exception as exc:
                    rerank_diag = {"enabled": True, "error": str(exc)}
            else:
                rerank_diag = {"enabled": False, "reason": "rerank_api_not_configured"}
        if rerank_diag:
            diagnostics["api_rerank"] = rerank_diag

        # ---- Build output ----
        results_out = ranked_pages[:top_k_pages] if top_k_pages > 0 else ranked_pages[:10]
        max_preview_chars = max(0, int(tool_defaults.FILE_SEARCH_SUMMARY_SNIPPET_PREVIEW_CHARS))
        max_snippets = max(1, int(tool_defaults.FILE_SEARCH_MAX_SNIPPETS_PER_FILE))
        preview_rows: List[Dict[str, Any]] = []
        for page, score in results_out:
            snips = page_snippets.get(page, [])
            snips.sort(key=lambda s: s.get("rank", 9999))
            shown: List[Dict[str, str]] = []
            for s in snips[:max_snippets]:
                text = s["snippet"]
                if max_preview_chars and len(text) > max_preview_chars:
                    text = text[:max(0, max_preview_chars - 3)].rstrip() + "..."
                shown.append({"channel": s["channel"], "snippet": text})
            preview_rows.append({
                "page": page, "score": round(score, 6),
                "sections": page_to_sections.get(page, []),
                "section_context": page_to_section_context.get(page, []),
                "snippets": shown,
            })

        # Suggested reads: top pages as a flat list of page numbers,
        # ready for read.pages(pages=[...]). Agent should pick top 1-3.
        suggested_reads: List[int] = sorted(p for p, _ in results_out[:5])

        answer = [{"page": p, "score": round(s, 6)} for p, s in results_out]
        summary = build_llm_envelope(
            thinking=f"Page-level locate within file {file_id}: {len(fused)} candidate pages from {len(channel_page_docscores)} channels.",
            answer=answer,
            extra={
                "file_id": file_id,
                "preview": preview_rows,
                "suggested_reads": suggested_reads,
                "next_steps": f"Use read.pages(file={file_id}, pages=[...]) on the top pages for citeable evidence.",
            },
        )
        diagnostics["results"] = answer
        diagnostics["suggested_reads"] = suggested_reads
        return ToolResult(summary=summary, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # API rerank (DashScope qwen3-rerank)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rerank_client():
        """Lazily build a DashScope rerank client from env-driven config."""
        from config.encapsulation.llm.rerank.dashscope import DashScopeRerankConfig

        cfg = DashScopeRerankConfig()
        client = cfg.build()
        if not client.is_configured():
            return None
        return client

    async def _api_rerank(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[str], Dict[str, Any]]:
        """Rerank file candidates using the DashScope rerank API.

        For each candidate file, concatenates its top snippets into a single
        "document" string, calls the rerank API, and returns the file_ids
        ordered by descending relevance_score.
        """
        diagnostics: Dict[str, Any] = {"method": "api_rerank", "candidate_count": len(candidates)}
        if not candidates:
            diagnostics["reason"] = "no_candidates"
            return [], diagnostics

        client = self._get_rerank_client()
        if client is None:
            diagnostics["reason"] = "rerank_api_not_configured"
            logger.debug("DashScope rerank API key not set; skipping API rerank.")
            return [], diagnostics

        diagnostics["model"] = client.model_name

        # Build document strings: concatenate top snippets per file candidate.
        # Sort by score (descending) across all channels so the most relevant
        # snippets are selected regardless of channel name.
        max_hits = max(1, int(tool_defaults.FILE_SEARCH_MAX_SNIPPETS_PER_FILE))
        documents: List[str] = []
        file_ids: List[str] = []
        for row in candidates:
            hit_items = list(row.get("hits") or [])
            hit_items.sort(key=lambda item: -float(item.get("score") or 0))
            snippets: List[str] = []
            for item in hit_items:
                if len(snippets) >= max_hits:
                    break
                snippet = str(item.get("snippet") or "").strip()
                if snippet:
                    snippets.append(snippet)
            filename = row.get("filename") or ""
            doc_text = f"[{filename}]\n" + "\n".join(snippets) if snippets else filename
            documents.append(doc_text)
            file_ids.append(str(row.get("file_id") or ""))

        _file_rerank_instruct = (
            f"User query is: {query}\n"
            "Select the document most likely to contain the answer to this query."
        )
        try:
            scored = await client.arerank(
                query=query, documents=documents, top_k=len(documents),
                instruct=_file_rerank_instruct,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics["error"] = str(exc)
            logger.warning("DashScope rerank API call failed: %s", exc)
            return [], diagnostics

        # Map (index, score) back to file_ids in score-descending order.
        ranked_file_ids: List[str] = []
        seen: set[str] = set()
        for idx, score in scored:
            if 0 <= idx < len(file_ids):
                fid = file_ids[idx]
                if fid and fid not in seen:
                    ranked_file_ids.append(fid)
                    seen.add(fid)

        diagnostics["ranked_file_ids"] = ranked_file_ids
        diagnostics["scores"] = [(idx, score) for idx, score in scored]
        return ranked_file_ids, diagnostics
