"""Unified locate tool: file-level routing via multi-channel retrieval."""
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from config.core.deepsearch import tool_defaults
from core.deepsearch.utils.file_scope import FileScope
from core.deepsearch.utils.ids import coerce_uuid_list
from core.prompts.deepsearch.search_file import (
    SEARCH_FILE_RERANK_SYSTEM_PROMPT_EN,
    SEARCH_FILE_RERANK_USER_PROMPT_TEMPLATE_EN,
)
from core.deepsearch.utils.llm_envelope import build_llm_envelope
from core.deepsearch.utils.llm_json import call_llm_json_with_retry

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from .search.base import _ChannelResult, _SearchToolBase
from .search.bm25 import _Bm25Channel
from .search.faiss import _FaissChannel
from .search.graph_chunk import _GraphChunkChannel
from .search.regex import _RegexChannel


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


@dataclass
class _Hit:
    channel: str
    chunk_id: str
    rank: int
    score: Optional[float]
    snippet: str


class LocateTool(_SearchToolBase, _FaissChannel, _Bm25Channel, _GraphChunkChannel, _RegexChannel, GraphTool):
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
    def _should_skip_llm_rerank(cls, *, query: str, candidates: Sequence[Mapping[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        """Return (skip, diagnostics) for the rerank LLM call."""

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
            "Find relevant files by running parallel chunk retrieval across 4 channels "
            "(dense, bm25, graph_chunk, regex), aggregating by file_id via RRF fusion. "
            "Uses QuerySpec (from initial_think) to seed search channels automatically. "
            "Returns ranked file candidates with snippets for routing decisions."
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
                "focus_query": {"type": "string", "description": "Optional query override."},
                "file": {"type": "string", "description": "Optional file_id to scope results to a single file."},
                "top_k": {"type": "integer", "minimum": 0, "description": "How many files to return."},
                "owner_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Owner ids to search (e.g. [me, share]).",
                },
            }
        ),
        example_args={
            "question": "Which PDF mentions multi-currency switching?",
            "plan_step": "plan_01",
            "extra": {"top_k": 5, "owner_ids": ["<me-owner-id>"]},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)

        query_spec: Dict[str, Any] = {}
        if request.graph_context and isinstance(request.graph_context.metadata, dict):
            raw = request.graph_context.metadata.get("query_spec")
            if isinstance(raw, dict):
                query_spec = dict(raw)

        hyde_query = str(query_spec.get("hyde_query") or "").strip()
        dense_query = hyde_query if hyde_query else query

        raw_bm25_terms = query_spec.get("bm25_terms") or []
        if not isinstance(raw_bm25_terms, (list, tuple)):
            raw_bm25_terms = []
        bm25_extra_terms = [str(x).strip() for x in raw_bm25_terms if str(x).strip()]

        raw_regex = query_spec.get("regex_patterns") or []
        if not isinstance(raw_regex, (list, tuple)):
            raw_regex = []
        regex_patterns = [str(x).strip() for x in raw_regex if str(x).strip()]

        visibility = self._resolve_owner_visibility(request)
        diagnostics: Dict[str, Any] = {
            "query": query,
            "dense_query": dense_query,
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
        file_arg = (request.extra or {}).get("file")
        if file_arg:
            file_ids_coerced, invalid = coerce_uuid_list([str(file_arg)])
            if file_ids_coerced:
                file_scope = FileScope(file_ids=frozenset(file_ids_coerced), filename_contains=(), source="locate_arg")
            diagnostics["file_scope"] = {"file": str(file_arg), "valid": bool(file_ids_coerced), "invalid": invalid}

        top_k_files = self._resolve_top_k((request.extra or {}).get("top_k"), tool_defaults.FILE_SEARCH_DEFAULT_TOP_K)
        top_k_chunks = int(tool_defaults.FILE_SEARCH_CHANNEL_TOP_K)

        dense_extra = dict(request.extra or {})
        dense_extra["focus_query"] = dense_query
        dense_request = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=dense_extra,
            graph_context=request.graph_context,
            coverage_metrics=request.coverage_metrics,
        )

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
        tasks.append(
            (
                "faiss",
                asyncio.create_task(
                    self._run_faiss(request=dense_request, query=dense_query, top_k=top_k_chunks, file_scope=file_scope)
                ),
            )
        )
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
                row["hits"].append(
                    _Hit(
                        channel=channel,
                        chunk_id=chunk_id,
                        rank=rank_i,
                        score=score_f,
                        snippet=snippet,
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
            (request.extra or {}).get("enable_llm_rerank"), tool_defaults.FILE_SEARCH_ENABLE_LLM_RERANK_DEFAULT
        )
        rerank_top_k = self._resolve_top_k((request.extra or {}).get("rerank_top_k"), tool_defaults.FILE_SEARCH_RERANK_TOP_K)
        if rerank_enabled and self.llm_connector is not None and all_results and rerank_top_k != 0:
            skip, skip_diag = self._should_skip_llm_rerank(query=query, candidates=all_results)
            if skip:
                rerank_diag = {"enabled": False, **skip_diag}
            else:
                rerank_diag = {"enabled": True, **skip_diag}
            limited = all_results[:rerank_top_k] if rerank_top_k > 0 else all_results
            if not skip:
                reranked_ids, llm_diag = await self._llm_rerank(query=query, candidates=limited)
                rerank_diag.update(llm_diag)
                if reranked_ids:
                    remaining = [row for row in all_results if row.get("file_id") not in reranked_ids]
                    order_map = {fid: idx for idx, fid in enumerate(reranked_ids)}
                    limited_sorted = sorted(
                        [row for row in all_results if row.get("file_id") in order_map],
                        key=lambda row: order_map.get(row.get("file_id"), 9999),
                    )
                    all_results = limited_sorted + remaining
        if rerank_diag:
            diagnostics["llm_rerank"] = rerank_diag

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
                    extra={"reason": reason, "next_step": "Revise query terms or broaden owner_ids; then rerun locate."},
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
                        "snippet": snippet,
                    }
                )
                shown += 1
            preview_rows.append(preview)

        next_steps = (
            "Pick the best file_id from answer[0], then use toc.tree/tree.root + tree.children/tree.open "
            "to locate relevant pages, and finally use read.pages for citeable evidence."
        )
        llm_rerank = diagnostics.get("llm_rerank") if isinstance(diagnostics, dict) else None
        rerank_thinking = None
        if isinstance(llm_rerank, dict):
            rerank_thinking = str(llm_rerank.get("thinking") or llm_rerank.get("reasoning") or "").strip() or None

        ranked_file_ids = [str(row.get("file_id")) for row in results_out if row.get("file_id")]
        summary = build_llm_envelope(
            thinking="Aggregate chunk hits into file candidates (routing only), optionally rerank by user intent.",
            answer=ranked_file_ids,
            extra={
                "preview": preview_rows,
                "rerank_thinking": rerank_thinking,
                "next_steps": next_steps,
            },
        )
        return ToolResult(summary=summary, diagnostics=diagnostics)

    async def _llm_rerank(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[str], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {"enabled": True, "candidate_count": len(candidates)}
        if not candidates:
            diagnostics["reason"] = "no_candidates"
            return [], diagnostics
        if self.llm_connector is None:
            diagnostics["reason"] = "llm_unavailable"
            return [], diagnostics

        max_hits = max(1, int(tool_defaults.FILE_SEARCH_MAX_SNIPPETS_PER_FILE))
        payload_candidates: List[Dict[str, Any]] = []
        for row in candidates:
            hits = list(row.get("hits") or [])
            hits.sort(key=lambda item: (str(item.get("channel") or ""), int(item.get("rank") or 9999)))
            snippets: List[str] = []
            for item in hits:
                if len(snippets) >= max_hits:
                    break
                snippet = str(item.get("snippet") or "").strip()
                if not snippet:
                    continue
                snippets.append(snippet)
            payload_candidates.append(
                {
                    "file_id": row.get("file_id"),
                    "filename": row.get("filename") or "",
                    "hit_count": int(row.get("hit_count") or 0),
                    "per_channel_hits": row.get("per_channel_hits") or {},
                    "snippets": snippets,
                }
            )

        user_payload = {
            "question": query,
            "candidates": payload_candidates,
        }
        messages = [
            {"role": "system", "content": SEARCH_FILE_RERANK_SYSTEM_PROMPT_EN},
            {
                "role": "user",
                "content": SEARCH_FILE_RERANK_USER_PROMPT_TEMPLATE_EN.format(
                    question=query,
                    payload=json.dumps(user_payload, ensure_ascii=True),
                ),
            },
        ]
        model_name = getattr(getattr(self.llm_connector, "config", None), "model_name", None)
        if model_name:
            diagnostics["model"] = model_name

        try:
            payload, raw_response = await call_llm_json_with_retry(
                llm_connector=self.llm_connector,
                messages=messages,
                expected="dict",
                temperature=float(tool_defaults.FILE_SEARCH_RERANK_TEMPERATURE),
                max_tokens=None,
                return_raw=True,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics["error"] = str(exc)
            return [], diagnostics

        raw_response = str(raw_response or "")
        diagnostics["raw_response"] = raw_response
        if not isinstance(payload, dict):
            diagnostics["error"] = "json_parse_failed"
            return [], diagnostics
        ranked = payload.get("answer")
        if ranked is None:
            ranked = payload.get("ranked_file_ids")
        if not isinstance(ranked, list):
            diagnostics["error"] = "missing_ranked_file_ids"
            return [], diagnostics

        candidate_ids = {row.get("file_id") for row in candidates if row.get("file_id")}
        filename_to_id: Dict[str, str] = {}
        for row in candidates:
            fid = row.get("file_id")
            name = row.get("filename")
            if not fid or not name:
                continue
            key = os.path.basename(str(name)).strip().lower()
            if key and key not in filename_to_id:
                filename_to_id[key] = fid
        normalized: List[str] = []
        for item in ranked:
            token = str(item or "").strip()
            if not token:
                continue
            if token in candidate_ids and token not in normalized:
                normalized.append(token)
                continue
            name_key = os.path.basename(token).strip().lower()
            mapped = filename_to_id.get(name_key)
            if mapped and mapped not in normalized:
                normalized.append(mapped)

        thinking = payload.get("thinking")
        if not isinstance(thinking, str) or not thinking.strip():
            thinking = payload.get("reasoning")
        if isinstance(thinking, str) and thinking.strip():
            diagnostics["thinking"] = thinking.strip()

        diagnostics["ranked_file_ids"] = normalized
        return normalized, diagnostics
