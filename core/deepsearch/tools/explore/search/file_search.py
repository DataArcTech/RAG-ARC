"""search.file tool: relevant-file routing via global chunk-level retrieval.

`search.file` answers: "Which file(s) should we scope to next?"

Behavior:
- runs chunk-level retrieval globally (faiss/bm25/graph_chunk) across allowed owner scopes
- aggregates hits by file_id (RRF over ranks)
- optionally reranks candidates with LLM using macro+micro signals
- returns candidate file_ids with short "why relevant" snippets for routing decisions
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from config.core.deepsearch import tool_defaults
from core.deepsearch.utils.owner_visibility import resolve_owner_visibility
from core.prompts.deepsearch.search_file import (
    SEARCH_FILE_RERANK_SYSTEM_PROMPT_EN,
    SEARCH_FILE_RERANK_USER_PROMPT_TEMPLATE_EN,
)
from core.deepsearch.utils.llm_envelope import build_llm_envelope

from ...base import (
    GraphTool,
    ToolDescriptor,
    ToolResult,
    ToolRunRequest,
    build_input_schema,
)
from ...governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER
from core.deepsearch.utils.llm_json import call_llm_json_with_retry
from .base import _SearchToolBase
from .bm25 import _Bm25Channel
from .faiss import _FaissChannel
from .graph_chunk import _GraphChunkChannel


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
    # Standard RRF: 1 / (k + rank). rank is 1-based.
    return 1.0 / float(max(1, int(k) + int(rank)))


@dataclass
class _Hit:
    channel: str
    chunk_id: str
    rank: int
    score: Optional[float]
    snippet: str


class FileSearchTool(_SearchToolBase, _FaissChannel, _Bm25Channel, _GraphChunkChannel, GraphTool):
    """Relevant-file routing tool (evidence-driven)."""

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

        # Skip condition: (top1 - top2) / top1 > threshold
        # (more conservative than dividing by top2; reduces accidental skips).
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
        name="search.file",
        channel="graph",
        description=(
            "Find relevant file candidates by running global chunk-level retrieval "
            "(faiss + bm25 + graph_chunk), aggregating hits by file_id, and returning "
            "candidate files with short 'why relevant' snippets. "
            "Optionally reranks candidates with LLM to align with user intent. "
            "Use this to pick file_id(s), then use tree/toc + read.pages for evidence exploration."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "file_search", "relevant_files", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.search.file",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "top_k": {"type": "integer", "minimum": 0, "description": "How many files to return."},
                "enable_llm_rerank": {"type": "boolean", "description": "Enable LLM rerank of file candidates."},
                "rerank_top_k": {"type": "integer", "minimum": 0, "description": "How many candidates to send to LLM."},
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Channels: faiss, bm25, graph_chunk (or 'graph' alias). Defaults to all.",
                },
                "owner_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Owner ids to search (e.g. [me, share]). Must be authorized by code-side whitelist.",
                },
                "faiss_top_k": {"type": "integer", "minimum": 0, "description": "Optional override of channel top-k."},
                "bm25_top_k": {"type": "integer", "minimum": 0, "description": "Optional override of channel top-k."},
                "graph_top_k": {"type": "integer", "minimum": 0, "description": "Optional override of channel top-k."},
                "use_ppr": {"type": "boolean", "description": "Only for graph_chunk channel."},
            }
        ),
        example_args={
            "question": "Which PDF mentions multi-currency switching?",
            "plan_step": "plan_01",
            "extra": {"top_k": 5, "channels": ["bm25", "faiss", "graph"], "owner_ids": ["<me-owner-id>"]},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)
        visibility = resolve_owner_visibility(
            extra=request.extra,
            access_scope=request.access_scope,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
        )
        diagnostics: Dict[str, Any] = {"query": query, "owner_visibility": visibility.as_dict()}

        if not visibility.enabled:
            return ToolResult(
                summary="search.file skipped: missing owner scope.",
                diagnostics={**diagnostics, "reason": "owner_scope_missing"},
            )

        # Which chunk-retrieval channels to use for routing.
        channels, unknown = self._resolve_channels(request.extra or {})
        diagnostics["channels"] = channels
        diagnostics["unknown_channels"] = unknown
        if not channels:
            return ToolResult(
                summary="search.file skipped: no channels selected.",
                diagnostics={**diagnostics, "reason": "channels_empty"},
            )

        # How many candidate files to return (final).
        top_k_files = self._resolve_top_k(request.extra.get("top_k"), tool_defaults.FILE_SEARCH_DEFAULT_TOP_K)
        # Retrieval depth for chunks (per channel). We rely on channel-specific overrides when present.
        top_k_chunks = int(tool_defaults.FILE_SEARCH_CHANNEL_TOP_K)

        # Run selected channels and collect chunk evidences for routing-only aggregation.
        channel_summaries: List[str] = []
        channel_diags: Dict[str, Any] = {}

        # File-scope is intentionally disabled for search.file (global routing).
        file_scope = None

        hits: Dict[str, Dict[str, Any]] = {}
        seen_chunk_ids_by_file: Dict[str, set[str]] = {}
        per_channel_scores: Dict[str, Dict[str, List[float]]] = {}
        rrf_k = int(tool_defaults.FILE_SEARCH_RRF_K)

        async def _consume(channel: str, evidences: Sequence[Any], *, diag: Mapping[str, Any]) -> None:
            channel_diags[channel] = dict(diag or {})
            for ev in evidences or []:
                provenance = getattr(ev, "provenance", None) or {}
                meta = provenance.get("metadata") if isinstance(provenance, dict) else None
                meta = dict(meta or {}) if isinstance(meta, dict) else {}
                file_id = _coerce_file_id(meta)
                if not file_id:
                    continue

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

        # faiss
        if "faiss" in channels:
            result = await self._run_faiss(request=request, query=query, top_k=top_k_chunks, file_scope=file_scope)
            channel_summaries.append(result.summary)
            await _consume("faiss", result.evidences, diag=result.diagnostics)

        # bm25
        if "bm25" in channels:
            result = await self._run_bm25(request=request, query=query, top_k=top_k_chunks, file_scope=file_scope)
            channel_summaries.append(result.summary)
            await _consume("bm25", result.evidences, diag=result.diagnostics)

        # graph_chunk
        if "graph_chunk" in channels:
            try:
                result = await self._run_graph_chunk(request=request, query=query, top_k=top_k_chunks, file_scope=file_scope)
                channel_summaries.append(result.summary)
                await _consume("graph_chunk", result.evidences, diag=result.diagnostics)
            except Exception as exc:  # noqa: BLE001
                # Keep this observable; graph channel may be unavailable in some deployments.
                channel_summaries.append("graph_chunk search failed.")
                channel_diags["graph_chunk"] = {"query": query, "error": str(exc)}

        diagnostics["channels_summary"] = channel_summaries
        diagnostics["channels_diagnostics"] = channel_diags

        # DocScore per channel: sum(scores) / sqrt(N+1)
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

        # Attach fused score to rows for ordering.
        for file_id, row in hits.items():
            row["score"] = float(fused_scores.get(file_id) or 0.0)
            row["per_channel_docscore"] = {ch: channel_doc_scores.get(ch, {}).get(file_id) for ch in channel_doc_scores}
            row["per_channel_docrank"] = {ch: channel_doc_ranks.get(ch, {}).get(file_id) for ch in channel_doc_ranks}

        diagnostics["channel_doc_scores"] = channel_doc_scores
        diagnostics["channel_doc_ranks"] = channel_doc_ranks

        all_results = sorted(hits.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)
        rerank_diag: Dict[str, Any] = {}
        rerank_enabled = self._coerce_bool(
            request.extra.get("enable_llm_rerank"), tool_defaults.FILE_SEARCH_ENABLE_LLM_RERANK_DEFAULT
        )
        rerank_top_k = self._resolve_top_k(request.extra.get("rerank_top_k"), tool_defaults.FILE_SEARCH_RERANK_TOP_K)
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

        results = all_results[:top_k_files] if top_k_files > 0 else []

        diagnostics["results"] = results

        if not results:
            reason = "empty_hit"
            if visibility.owner_ids_rejected:
                reason = "owner_ids_rejected"
            return ToolResult(
                summary=build_llm_envelope(
                    thinking="Global file routing via chunk retrieval produced no usable file candidates.",
                    answer=[],
                    extra={"reason": reason, "next_step": "Revise query terms or broaden owner_ids/channels; then rerun search.file."},
                ),
                diagnostics={**diagnostics, "reason": reason},
            )

        # Human-readable preview for routing decisions (embedded into the JSON envelope).
        max_hits = max(1, int(tool_defaults.FILE_SEARCH_MAX_SNIPPETS_PER_FILE))
        max_preview = max(0, int(tool_defaults.FILE_SEARCH_SUMMARY_SNIPPET_PREVIEW_CHARS))
        preview_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(results[: min(len(results), 5)]):
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

        ranked_file_ids = [str(row.get("file_id")) for row in results if row.get("file_id")]
        summary = build_llm_envelope(
            thinking="Aggregate global chunk hits into file candidates (routing only), optionally rerank by user intent.",
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
