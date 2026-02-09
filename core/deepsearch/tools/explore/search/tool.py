"""Multi-channel search tool orchestrator."""
import asyncio
from typing import Any, Dict, List, Optional

from config.core.deepsearch import tool_defaults
from config import pageindex as pageindex_cfg
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.file_scope import resolve_file_scope
from core.deepsearch.utils.ids import coerce_uuid_list

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from .base import _SearchToolBase, strip_file_scope_from_graph_context
from .bm25 import _Bm25Channel
from .faiss import _FaissChannel
from .graph_chunk import _GraphChunkChannel


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _suggest_read_pages_from_evidences(evidences: List[EvidenceChunk]) -> List[Dict[str, Any]]:
    """Suggest read.pages ranges based on evidence provenance metadata."""

    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    max_items = max(0, int(tool_defaults.SEARCH_SUGGESTED_READ_MAX))
    for ev in evidences:
        if max_items and len(out) >= max_items:
            break
        prov = getattr(ev, "provenance", None) or {}
        meta = prov.get("metadata") if isinstance(prov, dict) else None
        if not isinstance(meta, dict):
            continue
        file_id = str(meta.get("source_file_id") or meta.get("file_id") or "").strip()
        ps = _coerce_int(meta.get("section_page_start"))
        if ps is None:
            ps = _coerce_int(meta.get("page_start"))
        pe = _coerce_int(meta.get("section_page_end"))
        if pe is None:
            pe = _coerce_int(meta.get("page_end"))
        if not file_id or ps is None or pe is None:
            continue
        if pe < ps:
            ps, pe = pe, ps
        key = (file_id, ps, pe)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "tool": "read.pages",
                "args": {"file_id": file_id, "page_start": ps, "page_end": pe},
                "why": "expand_full_context_for_top_hit",
            }
        )
    return out


def _collect_term_hints(evidences: List[EvidenceChunk]) -> List[str]:
    """Collect document-native term hints from section paths/titles."""
    max_items = max(0, int(tool_defaults.SEARCH_TERM_HINTS_MAX))
    if max_items <= 0:
        return []
    delimiter = pageindex_cfg.section_path_delimiter()
    seen: set[str] = set()
    out: List[str] = []
    for ev in evidences:
        prov = getattr(ev, "provenance", None) or {}
        meta = prov.get("metadata") if isinstance(prov, dict) else None
        if not isinstance(meta, dict):
            continue
        for key in ("section_title", "section_path", "title", "path"):
            raw = str(meta.get(key) or "").strip()
            if not raw:
                continue
            candidates = [raw]
            if key == "section_path" or key == "path":
                candidates = [seg.strip() for seg in raw.split(delimiter) if seg.strip()]
            for item in candidates:
                if not item or item in seen:
                    continue
                seen.add(item)
                out.append(item)
                if len(out) >= max_items:
                    return out
    return out


class SearchTool(_SearchToolBase, _FaissChannel, _Bm25Channel, _GraphChunkChannel, GraphTool):
    """Search across faiss, bm25, and graph_chunk channels."""

    descriptor = ToolDescriptor(
        name="search",
        channel="graph",
        description=(
            "Three-way fast search (faiss + bm25 + graph_chunk) for quick localization. "
            "Evidence: derived routing snippets; citeable evidence comes from read.pages. "
            "Good: focus_query='HippoRAG graph', channels=['faiss','bm25'], top_k=10. "
            "Bad: empty query or repeating the same args after empty_hit."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "faiss", "bm25", "graph_chunk", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.search",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "file_id": {"type": "string", "description": "Restrict results to a specific file_id."},
                "file_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict results to a set of file_ids.",
                },
                "filename_contains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict results by filename substring (best-effort).",
                },
                "section_id": {"type": "string", "description": "Restrict results to a specific section_id."},
                "section_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict results to a set of section_ids (exact match).",
                },
                "owner_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Owner ids to search (e.g. [me, share]). Must be authorized by code-side whitelist.",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Channels to execute: faiss, bm25, graph_chunk (default: all).",
                },
                "top_k": {"type": "integer", "minimum": 0, "description": "Default top-k for all channels."},
                "faiss_top_k": {"type": "integer", "minimum": 0, "description": "Top-k override for faiss."},
                "bm25_top_k": {"type": "integer", "minimum": 0, "description": "Top-k override for bm25."},
                "graph_top_k": {"type": "integer", "minimum": 0, "description": "Top-k override for graph_chunk."},
                "use_ppr": {"type": "boolean", "description": "Enable PPR for graph_chunk (default off)."},
                "enable_llm_rerank": {"type": "boolean", "description": "Enable LLM fact rerank for graph_chunk."},
                "enable_entity_fallback": {"type": "boolean", "description": "Enable entity fallback for graph_chunk."},
                "entity_seed_top_k": {"type": "integer", "minimum": 0, "description": "Max entities from fallback."},
                "seed_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional seed entities (entity_id or name).",
                },
                "bm25_use_phrase_query": {"type": "boolean", "description": "Use phrase query for BM25."},
                "graph_overrides": {
                    "type": "object",
                    "description": "Graph retrieval overrides (allowlisted safe keys only).",
                },
            }
        ),
        example_args={
            "question": "HippoRAG graph retrieval",
            "plan_step": "plan_01",
            "extra": {"channels": ["faiss", "bm25", "graph_chunk"], "top_k": 10},
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
        channels, unknown = self._resolve_channels(request.extra or {})
        if not channels:
            return ToolResult(
                summary="Search skipped: no channels selected.",
                diagnostics={"query": query, "channels": [], "unknown_channels": unknown},
            )

        tasks: List[asyncio.Task] = []
        for channel in channels:
            if channel == "faiss":
                tasks.append(asyncio.create_task(self._run_faiss(request=request, query=query, top_k=top_k, file_scope=file_scope)))
            elif channel == "bm25":
                tasks.append(asyncio.create_task(self._run_bm25(request=request, query=query, top_k=top_k, file_scope=file_scope)))
            elif channel == "graph_chunk":
                tasks.append(asyncio.create_task(self._run_graph_chunk(request=request, query=query, top_k=top_k, file_scope=file_scope)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors: List[str] = []
        evidences: List[EvidenceChunk] = []
        windows: Dict[str, Any] = {}
        summaries: List[str] = []
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue
            evidences.extend(result.evidences)
            windows[result.channel] = result.diagnostics
            summaries.append(result.summary)

        if not evidences and errors:
            raise RuntimeError(f"search failed: {errors[0]}")

        summary = "Search completed." + (" " + " ".join(summaries) if summaries else "")
        diagnostics = {
            "query": query,
            "channels": channels,
            "unknown_channels": unknown,
            "errors": errors,
            "windows": windows,
        }
        term_hints = _collect_term_hints(evidences)
        if term_hints:
            diagnostics["term_hints"] = term_hints
            summary = summary.rstrip() + " TERM_HINTS: " + ", ".join(term_hints) + "."
        suggested_reads = _suggest_read_pages_from_evidences(evidences)
        if suggested_reads:
            diagnostics["suggested_reads"] = suggested_reads
            # Keep summary short but actionable.
            tips = []
            for item in suggested_reads[: max(1, int(tool_defaults.SEARCH_SUGGESTED_READ_MAX))]:
                args = item.get("args") if isinstance(item, dict) else None
                if not isinstance(args, dict):
                    continue
                tips.append(f"read.pages(file_id={args.get('file_id')}, pages={args.get('page_start')}-{args.get('page_end')})")
            if tips:
                summary = summary.rstrip() + " TIP: consider " + "; ".join(tips) + " for full context."
        elif not evidences and getattr(file_scope, "file_ids", None):
            # No hits: suggest a small overview read for the scoped file(s).
            page_end = int(tool_defaults.SEARCH_EMPTY_SUGGEST_OVERVIEW_PAGE_END)
            if page_end >= 0:
                file_ids = list(getattr(file_scope, "file_ids") or [])[:1]
                if file_ids:
                    diagnostics["suggested_reads"] = [
                        {
                            "tool": "read.pages",
                            "args": {"file_id": file_ids[0], "page_start": 0, "page_end": page_end},
                            "why": "no_hits_read_overview_pages",
                        }
                    ]
                    summary = summary.rstrip() + f" TIP: no hits; consider read.pages(file_id={file_ids[0]}, pages=0-{page_end}) to inspect key facts/terms."
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)


def _coerce_file_ids(extra: Dict[str, Any] | None) -> List[str]:
    payload = extra or {}
    out: List[str] = []
    seen: set[str] = set()
    for key in ("file_ids", "file_id", "source_file_ids", "source_file_id"):
        raw = payload.get(key)
        if raw is None:
            continue
        items = raw if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
        for item in items:
            token = str(item or "").strip()
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


class SearchScopedTool(SearchTool):
    """Scoped multi-channel search: requires an explicit file scope (file_id/file_ids)."""

    descriptor = ToolDescriptor(
        name="search.scoped",
        channel="graph",
        description=(
            "Scoped three-way search (faiss + bm25 + graph_chunk). "
            "Requires explicit file_id/file_ids to prevent cross-document/company contamination. "
            "Use search.file first to choose the file, then use this tool."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "scoped", "faiss", "bm25", "graph_chunk", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.search.scoped",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "file_id": {"type": "string", "description": "Restrict results to a specific file_id (required unless file_ids provided)."},
                "file_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to file_ids (required unless file_id provided)."},
                "section_id": {"type": "string", "description": "Restrict results to a specific section_id."},
                "section_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to section_ids."},
                "channels": {"type": "array", "items": {"type": "string"}, "description": "Channels: faiss, bm25, graph_chunk."},
                "top_k": {"type": "integer", "minimum": 0, "description": "Default top-k for all channels."},
            }
        ),
        example_args={
            "question": "Find relevant chunks in a specific file",
            "plan_step": "plan_01",
            "extra": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "channels": ["faiss", "bm25", "graph_chunk"], "top_k": 10},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)
        file_ids_raw = _coerce_file_ids(request.extra)
        scope_source = "tool_args"
        if not file_ids_raw:
            try:
                meta = request.graph_context.metadata if request.graph_context and isinstance(request.graph_context.metadata, dict) else {}
                scope = meta.get("file_scope") if isinstance(meta, dict) else None
                if isinstance(scope, dict):
                    inherited = scope.get("file_ids") or scope.get("source_file_ids") or []
                    if isinstance(inherited, (list, tuple, set, frozenset)):
                        file_ids_raw = [str(x) for x in inherited]
                        if file_ids_raw:
                            scope_source = "graph_context"
            except Exception:
                file_ids_raw = file_ids_raw
        file_ids, invalid = coerce_uuid_list(file_ids_raw)
        if invalid:
            return ToolResult(
                summary="search.scoped skipped: invalid file_id(s) (expected UUID; use search.file to obtain file_id).",
                diagnostics={"reason": "invalid_file_id_format", "query": query, "invalid_file_ids": invalid[:20]},
            )
        if not file_ids:
            return ToolResult(
                summary="search.scoped skipped: missing file_id/file_ids (call search.file first).",
                diagnostics={"reason": "missing_file_scope", "query": query, "required": ["file_id", "file_ids"]},
            )

        patched_extra = dict(request.extra or {})
        # Normalize into UUIDs to keep tool traces stable and prevent accidental broadening.
        patched_extra.pop("filename_contains", None)
        patched_extra.pop("source_file_name", None)
        patched_extra.pop("source_file_names", None)
        patched_extra["file_ids"] = list(file_ids)
        if len(file_ids) == 1:
            patched_extra["file_id"] = file_ids[0]

        patched = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=patched_extra,
            graph_context=request.graph_context,
            coverage_metrics=request.coverage_metrics,
        )
        result = await super().run(patched)
        diag = dict(result.diagnostics or {})
        diag.setdefault("file_scope_source", scope_source)
        result.diagnostics = diag
        return result


class SearchGlobalTool(SearchTool):
    """Global multi-channel search: intentionally ignores inherited file_scope from graph_context."""

    descriptor = ToolDescriptor(
        name="search.global",
        channel="graph",
        description=(
            "Global three-way search (faiss + bm25 + graph_chunk) across all accessible documents. "
            "Warning: may introduce cross-document/company noise. Prefer search.scoped whenever possible."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("search", "global", "faiss", "bm25", "graph_chunk", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.search.global",
        mcp_callable=True,
        input_schema=SearchTool.descriptor.input_schema,
        example_args={
            "question": "Find relevant chunks across all documents",
            "plan_step": "plan_01",
            "extra": {"channels": ["faiss", "bm25", "graph"], "top_k": 10},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        # Explicitly ignore inherited file_scope from graph_context.metadata to avoid
        # accidental "scope persistence" when the user wants a true global expansion.
        extra = dict(request.extra or {})
        # Also ignore explicit file filters to keep semantics stable: global means global.
        for key in ("file_id", "file_ids", "source_file_id", "source_file_ids", "filename_contains"):
            extra.pop(key, None)
        patched = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=extra,
            graph_context=strip_file_scope_from_graph_context(request.graph_context),
            coverage_metrics=request.coverage_metrics,
        )
        result = await super().run(patched)
        risk = "global_search_may_introduce_cross_doc_noise"
        result.diagnostics = {**dict(result.diagnostics or {}), "search_mode": "global", "risk": risk}
        if result.summary:
            result.summary = result.summary.rstrip() + f" NOTE: {risk}."
        else:
            result.summary = f"Search completed. NOTE: {risk}."
        return result
