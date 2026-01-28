"""file.search tool: doc-level routing using PageIndex doc description index.

This is intentionally *not* full-text search. It returns candidate `file_id`s plus
their PageIndex-generated `doc_description`, used for doc routing and subsequent
file-scoped chunk retrieval.
"""

from typing import Any, Dict, List, Optional

from config import pageindex as pageindex_cfg
from config.core.deepsearch import tool_defaults
from core.deepsearch.utils.owner_visibility import resolve_owner_visibility
from core.retrieval.pageindex_retriever import PageIndexRetriever

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER


def _coerce_file_id(meta: Any, fallback: Optional[str]) -> Optional[str]:
    if isinstance(meta, dict):
        for key in ("source_file_id", "file_id", "doc_id", "document_id"):
            token = str(meta.get(key) or "").strip()
            if token:
                return token
    token = str(fallback or "").strip()
    return token or None


def _split_title_desc(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    lines = [line.rstrip() for line in raw.splitlines()]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return "", ""
    title = lines[0].strip()
    desc = "\n".join(lines[1:]).strip()
    return title, desc


class FileSearchTool(GraphTool):
    """Doc-level file routing tool (PageIndex doc_routing)."""

    descriptor = ToolDescriptor(
        name="file.search",
        channel="graph",
        description=(
            "Doc-level routing search using PageIndex doc descriptions (NOT full-text). "
            "Returns candidate file_ids with filename/title/description (+ optional doc_profile) "
            "for subsequent file-scoped search."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "file_search", "doc_routing", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.file_search",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "top_k": {"type": "integer", "minimum": 0, "description": "How many files to return."},
                "owner_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Owner ids to search (e.g. [me, share]). Must be authorized by code-side whitelist.",
                },
            }
        ),
        example_args={
            "question": "Find the file about warranty terms",
            "plan_step": "plan_01",
            "extra": {"top_k": 3, "owner_ids": ["<me-owner-id>", "<share-owner-id>"]},
        },
    )

    def __init__(self, *, pageindex_retriever: PageIndexRetriever | None = None) -> None:
        self._pageindex = pageindex_retriever or (PageIndexRetriever() if pageindex_cfg.pageindex_enabled() else None)

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)
        visibility = resolve_owner_visibility(
            extra=request.extra,
            access_scope=request.access_scope,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
        )
        diagnostics: Dict[str, Any] = {"query": query, "owner_visibility": visibility.as_dict()}

        if not pageindex_cfg.pageindex_enabled() or not pageindex_cfg.doc_routing_enabled():
            return ToolResult(
                summary="file.search skipped: PageIndex doc routing is disabled.",
                diagnostics={**diagnostics, "reason": "doc_routing_disabled"},
            )
        if self._pageindex is None:
            return ToolResult(
                summary="file.search skipped: PageIndex retriever unavailable.",
                diagnostics={**diagnostics, "reason": "pageindex_retriever_unavailable"},
            )
        if not visibility.enabled:
            return ToolResult(
                summary="file.search skipped: missing owner scope.",
                diagnostics={**diagnostics, "reason": "owner_scope_missing"},
            )

        top_k = self._resolve_top_k(request.extra.get("top_k"), tool_defaults.FILE_SEARCH_DEFAULT_TOP_K)
        cand_k = int(tool_defaults.FILE_SEARCH_RETRIEVE_CANDIDATES_K)
        max_desc_chars = max(120, int(tool_defaults.FILE_SEARCH_DESC_MAX_CHARS))

        # Gather per-owner results (me + share) then merge by best score per file_id.
        merged: Dict[str, Dict[str, Any]] = {}
        per_owner_counts: Dict[str, int] = {}
        for owner_id in visibility.owner_ids_used:
            try:
                hits = self._pageindex.retrieve_doc_chunks(query, owner_id=owner_id, k_final=top_k, k_candidates=cand_k)
            except Exception as exc:  # noqa: BLE001
                diagnostics.setdefault("errors", []).append({"owner_id": owner_id, "error": str(exc)})
                continue
            per_owner_counts[owner_id] = len(hits)
            for hit in hits:
                meta = getattr(hit, "metadata", None) or {}
                file_id = _coerce_file_id(meta, getattr(hit, "id", None))
                if not file_id:
                    continue
                score = meta.get("score")
                score_f = float(score) if isinstance(score, (int, float)) else 0.0
                filename = str(meta.get("filename") or meta.get("source_file_name") or "").strip()
                title, desc = _split_title_desc(getattr(hit, "content", ""))
                doc_desc = (desc or "").strip()
                if len(doc_desc) > max_desc_chars:
                    doc_desc = doc_desc[: max_desc_chars - 3].rstrip() + "..."
                doc_profile = {
                    "company": str(meta.get("doc_profile_company") or "").strip() or None,
                    "product": str(meta.get("doc_profile_product") or "").strip() or None,
                    "model": str(meta.get("doc_profile_model") or "").strip() or None,
                    "version": str(meta.get("doc_profile_version") or "").strip() or None,
                    "doc_type": str(meta.get("doc_profile_doc_type") or "").strip() or None,
                    "language": str(meta.get("doc_profile_language") or "").strip() or None,
                    "keywords": meta.get("doc_profile_keywords") if isinstance(meta.get("doc_profile_keywords"), list) else None,
                    "aliases": meta.get("doc_profile_aliases") if isinstance(meta.get("doc_profile_aliases"), list) else None,
                }
                # Trim empty profiles to keep payload compact.
                if not any(v for v in doc_profile.values()):
                    doc_profile = None
                existing = merged.get(file_id)
                if existing is None or score_f > float(existing.get("score") or 0.0):
                    merged[file_id] = {
                        "file_id": file_id,
                        "score": score_f,
                        "owner_id": owner_id,
                        "filename": filename or None,
                        "title": title or None,
                        "doc_description": doc_desc or None,
                        "doc_profile": doc_profile,
                    }

        results = sorted(merged.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)[:top_k]
        diagnostics["per_owner_retrieved"] = per_owner_counts
        diagnostics["results"] = results

        if not results:
            reason = "empty_hit"
            if visibility.owner_ids_rejected:
                reason = "owner_ids_rejected"
            return ToolResult(
                summary="file.search returned no files.",
                diagnostics={**diagnostics, "reason": reason},
            )

        # Put doc_description/doc_profile into the human-readable summary (not only diagnostics)
        # so the LLM can actually perform "doc routing" decisions.
        lines: List[str] = []
        for idx, row in enumerate(results[: min(len(results), 5)]):
            profile = row.get("doc_profile") if isinstance(row.get("doc_profile"), dict) else {}
            company = str(profile.get("company") or "").strip()
            version = str(profile.get("version") or "").strip()
            product = str(profile.get("product") or "").strip()
            suffix = ""
            if company or product or version:
                suffix = f" company={company} product={product} version={version}".rstrip()
            desc = str(row.get("doc_description") or "").strip()
            max_preview = int(tool_defaults.FILE_SEARCH_SUMMARY_DESC_PREVIEW_CHARS)
            if max_preview > 0 and len(desc) > max_preview:
                desc = desc[: max(0, max_preview - 3)].rstrip() + "..."
            lines.append(
                f"{idx+1}. file_id={row.get('file_id')} score={row.get('score'):.4f} owner_id={row.get('owner_id')} "
                f"filename={row.get('filename') or ''} title={row.get('title') or ''}{suffix}"
            )
            if desc:
                lines.append(f"   desc: {desc}")
        summary = "file.search returned candidate files:\n" + "\n".join(lines)
        return ToolResult(summary=summary, diagnostics=diagnostics)

    @staticmethod
    def _resolve_query(request: ToolRunRequest) -> str:
        extra = request.extra or {}
        focus = extra.get("focus_query") or extra.get("query") or request.question
        return str(focus or "").strip()

    @staticmethod
    def _resolve_top_k(value: Any, default: int) -> int:
        try:
            parsed = int(value) if value is not None else int(default)
        except Exception:
            parsed = int(default)
        return max(0, parsed)
