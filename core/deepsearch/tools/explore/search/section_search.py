"""section.search tool: section-level routing using PageIndex section index.

This is intentionally not full-text retrieval. It returns candidate sections (path + summary)
so the LLM can pick the right section before doing file-scoped chunk retrieval.
"""

from typing import Any, Dict, List

from config import pageindex as pageindex_cfg
from config.core.deepsearch import tool_defaults
from core.deepsearch.utils.owner_visibility import resolve_owner_visibility
from core.retrieval.pageindex_retriever import PageIndexRetriever

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER


def _coerce_section_id(meta: Any, fallback: Any) -> str | None:
    if isinstance(meta, dict):
        token = str(meta.get("section_id") or "").strip()
        if token:
            return token
    token = str(fallback or "").strip()
    return token or None


class SectionSearchTool(GraphTool):
    descriptor = ToolDescriptor(
        name="section.search",
        channel="graph",
        description=(
            "Section-level routing search using PageIndex section index (path + summary). "
            "Use this to find the right section_id/section_path within a file before chunk retrieval."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "section_search", "pageindex", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.section_search",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "top_k": {"type": "integer", "minimum": 0, "description": "How many sections to return."},
                "file_id": {"type": "string", "description": "Optional file_id restriction (recommended)."},
                "file_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional file_ids restriction."},
                "owner_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Owner ids to search (e.g. [me, share]). Must be authorized by code-side whitelist.",
                },
            }
        ),
        example_args={
            "question": "Find the installation requirements section",
            "plan_step": "plan_01",
            "extra": {"file_id": "<file_id>", "top_k": 5},
        },
    )

    def __init__(self, *, pageindex_retriever: PageIndexRetriever | None = None) -> None:
        self._pageindex = pageindex_retriever or (PageIndexRetriever() if pageindex_cfg.pageindex_enabled() else None)

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = str((request.extra or {}).get("focus_query") or (request.extra or {}).get("query") or request.question or "").strip()
        visibility = resolve_owner_visibility(
            extra=request.extra,
            access_scope=request.access_scope,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
        )
        diagnostics: Dict[str, Any] = {"query": query, "owner_visibility": visibility.as_dict()}

        if not pageindex_cfg.pageindex_enabled() or not pageindex_cfg.section_index_enabled():
            return ToolResult(
                summary="section.search skipped: PageIndex section index is disabled.",
                diagnostics={**diagnostics, "reason": "section_index_disabled"},
            )
        if self._pageindex is None:
            return ToolResult(
                summary="section.search skipped: PageIndex retriever unavailable.",
                diagnostics={**diagnostics, "reason": "pageindex_retriever_unavailable"},
            )
        if not visibility.enabled:
            return ToolResult(
                summary="section.search skipped: missing owner scope.",
                diagnostics={**diagnostics, "reason": "owner_scope_missing"},
            )
        if not query:
            return ToolResult(
                summary="section.search skipped: empty query.",
                diagnostics={**diagnostics, "reason": "empty_query"},
            )

        extra = request.extra or {}
        top_k = int(extra.get("top_k") or tool_defaults.SECTION_SEARCH_DEFAULT_TOP_K)
        top_k = max(0, top_k)

        file_ids: List[str] = []
        for key in ("file_ids", "file_id", "source_file_ids", "source_file_id"):
            raw = extra.get(key)
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, set, frozenset)):
                items = raw
            else:
                items = [raw]
            for item in items:
                token = str(item or "").strip()
                if token and token not in file_ids:
                    file_ids.append(token)

        merged: Dict[str, Dict[str, Any]] = {}
        per_owner_counts: Dict[str, int] = {}
        for owner_id in visibility.owner_ids_used:
            try:
                hits = self._pageindex.retrieve_sections(query, owner_id=owner_id, file_ids=file_ids or None)
            except Exception as exc:  # noqa: BLE001
                diagnostics.setdefault("errors", []).append({"owner_id": owner_id, "error": str(exc)})
                continue
            per_owner_counts[owner_id] = len(hits)
            for hit in hits:
                meta = getattr(hit, "metadata", None) or {}
                section_id = _coerce_section_id(meta, getattr(hit, "id", None))
                if not section_id:
                    continue
                score = meta.get("score")
                score_f = float(score) if isinstance(score, (int, float)) else 0.0
                row = {
                    "section_id": section_id,
                    "score": score_f,
                    "owner_id": owner_id,
                    "file_id": str(meta.get("source_file_id") or meta.get("file_id") or "").strip() or None,
                    "filename": str(meta.get("filename") or "").strip() or None,
                    "section_path": str(meta.get("section_path") or "").strip() or None,
                    "title": str(meta.get("title") or "").strip() or None,
                    "summary": str(meta.get("summary") or "").strip() or None,
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                }
                existing = merged.get(section_id)
                if existing is None or score_f > float(existing.get("score") or 0.0):
                    merged[section_id] = row

        results = sorted(merged.values(), key=lambda r: float(r.get("score") or 0.0), reverse=True)[:top_k]
        diagnostics["per_owner_retrieved"] = per_owner_counts
        diagnostics["file_ids_filter"] = file_ids
        diagnostics["results"] = results

        if not results:
            return ToolResult(summary="section.search returned no sections.", diagnostics={**diagnostics, "reason": "empty_hit"})

        lines: List[str] = []
        for idx, row in enumerate(results[: min(len(results), 6)]):
            lines.append(
                f"{idx+1}. section_id={row.get('section_id')} score={row.get('score'):.4f} "
                f"file_id={row.get('file_id') or ''} path={row.get('section_path') or ''}"
            )
        return ToolResult(summary="section.search returned candidate sections:\n" + "\n".join(lines), diagnostics=diagnostics)

