"""search.file tool: relevant-file routing via global chunk-level retrieval.

`search.file` answers: "Which file(s) should we scope to next?"

Behavior:
- runs chunk-level retrieval globally (faiss/bm25/graph_chunk) across allowed owner scopes
- aggregates hits by file_id (RRF over ranks)
- returns candidate file_ids with short "why relevant" snippets for routing decisions
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from config.core.deepsearch import tool_defaults
from core.deepsearch.utils.owner_visibility import resolve_owner_visibility

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER
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

    descriptor = ToolDescriptor(
        name="search.file",
        channel="graph",
        description=(
            "Find relevant file candidates by running global chunk-level retrieval "
            "(faiss + bm25 + graph_chunk), aggregating hits by file_id, and returning "
            "candidate files with short 'why relevant' snippets. "
            "Use this to pick file_id(s), then use search.scoped.* for precise evidence exploration."
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

        # Run selected channels and collect chunk evidences. We intentionally do not return evidences
        # from this tool; it is only for routing.
        channel_summaries: List[str] = []
        channel_diags: Dict[str, Any] = {}

        # File-scope is intentionally disabled for search.file (global routing).
        file_scope = None

        hits: Dict[str, Dict[str, Any]] = {}
        seen_chunk_ids_by_file: Dict[str, set[str]] = {}
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
                    row = {
                        "file_id": file_id,
                        "filename": _coerce_filename(meta),
                        "score": 0.0,
                        "hit_count": 0,
                        "hits": [],
                        "per_channel_hits": {},
                    }
                    hits[file_id] = row

                row["score"] = float(row.get("score") or 0.0) + _rrf(rank_i, k=rrf_k)
                row["hit_count"] = int(row.get("hit_count") or 0) + 1
                per_channel = dict(row.get("per_channel_hits") or {})
                per_channel[channel] = int(per_channel.get(channel) or 0) + 1
                row["per_channel_hits"] = per_channel
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

        results = sorted(hits.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)
        results = results[:top_k_files] if top_k_files > 0 else []

        diagnostics["results"] = results

        if not results:
            reason = "empty_hit"
            if visibility.owner_ids_rejected:
                reason = "owner_ids_rejected"
            return ToolResult(
                summary="search.file returned no candidate files.",
                diagnostics={**diagnostics, "reason": reason},
            )

        # Human-readable summary for routing decisions.
        max_hits = max(1, int(tool_defaults.FILE_SEARCH_MAX_SNIPPETS_PER_FILE))
        max_preview = max(0, int(tool_defaults.FILE_SEARCH_SUMMARY_SNIPPET_PREVIEW_CHARS))
        lines: List[str] = []
        for idx, row in enumerate(results[: min(len(results), 5)]):
            lines.append(
                f"{idx+1}. file_id={row.get('file_id')} score={float(row.get('score') or 0.0):.4f} "  # noqa: E501
                f"hits={int(row.get('hit_count') or 0)} filename={row.get('filename') or ''}"
            )
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
                channel = str(item.get("channel") or "")
                rank = int(item.get("rank") or 0)
                lines.append(f"   [{channel} rank={rank}] {snippet}")
                shown += 1

        summary = "search.file returned candidate files:\n" + "\n".join(lines)
        return ToolResult(summary=summary, diagnostics=diagnostics)
