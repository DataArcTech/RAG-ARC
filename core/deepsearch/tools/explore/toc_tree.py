"""toc.tree tool: list a file's section tree (PageIndex navigation).

Structure-first navigation: reads PageIndex-aligned Section nodes to reconstruct
a readable ToC tree for the LLM to pick a section before reading.
"""
from typing import Any, Dict, List

from config.core.deepsearch import tool_defaults
from config.core.deepsearch import runtime_cache_defaults
from core.deepsearch.utils.section_tree import (
    build_section_tree,
    fetch_section_nodes,
    fetch_section_tree_fingerprint,
    normalize_file_id,
)
from framework.cache import TTLRUCache

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


class TocTreeTool(GraphTool):
    descriptor = ToolDescriptor(
        name="toc.tree",
        channel="graph",
        description=(
            "Show the Table of Contents (section hierarchy) of a file. "
            "Returns an indented tree of section headings with page ranges. "
            "Use this to understand document structure before drilling into specific pages."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("toc", "pageindex", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.toc_tree",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "The file_id (UUID) to show the ToC for."},
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum heading depth to display (e.g. 2 shows only L1 and L2 sections). Defaults to all levels.",
                },
                "max_nodes": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of sections to display before truncating. Defaults to 200.",
                },
            },
            required_extra_fields=("file_id",),
        ),
        example_args={
            "question": "Show ToC for the manual",
            "plan_step": "plan_01",
            "extra": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "max_depth": 4},
        },
    )

    def __init__(
        self,
        *,
        cache_enabled: bool | None = None,
        cache_max_entries: int | None = None,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        if cache_enabled is None:
            cache_enabled = runtime_cache_defaults.DEFAULT_TOC_TREE_CACHE_ENABLED
        if cache_max_entries is None:
            cache_max_entries = runtime_cache_defaults.DEFAULT_TOC_TREE_CACHE_MAX_ENTRIES
        if cache_ttl_seconds is None:
            cache_ttl_seconds = float(runtime_cache_defaults.DEFAULT_TOC_TREE_CACHE_TTL_SECONDS)
        self._cache: TTLRUCache[tuple[str, str, str, int, int], Dict[str, Any]] | None = None
        if cache_enabled and cache_max_entries and cache_ttl_seconds:
            try:
                self._cache = TTLRUCache(max_entries=int(cache_max_entries), ttl_seconds=float(cache_ttl_seconds))
            except Exception:
                self._cache = None

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        file_id, file_id_raw = normalize_file_id(extra.get("file_id"))
        if not file_id:
            reason = "missing_file_id" if not file_id_raw else "invalid_file_id_format"
            return ToolResult(
                summary="toc.tree skipped: invalid/missing file_id (expected UUID; use locate to obtain file_id).",
                diagnostics={"reason": reason, "file_id_raw": file_id_raw or None},
            )

        max_depth = _coerce_int(extra.get("max_depth")) or int(tool_defaults.TOC_TREE_DEFAULT_MAX_DEPTH)
        max_nodes = _coerce_int(extra.get("max_nodes")) or int(tool_defaults.TOC_TREE_MAX_NODES)
        max_depth = max(1, max_depth)
        max_nodes = max(1, max_nodes)

        # Cache ToC trees by owner+file+tree fingerprint; formatting params are part of the key.
        if self._cache is not None and request.access_scope is not None:
            try:
                owner_scope_id = str(getattr(request.access_scope, "scope_id", "") or "")
            except Exception:
                owner_scope_id = ""
            if owner_scope_id:
                fp, fp_diag = await fetch_section_tree_fingerprint(
                    adapter=request.adapter,
                    access_scope=request.access_scope,
                    file_id=file_id,
                )
                if fp:
                    key = (owner_scope_id, file_id, fp, int(max_depth), int(max_nodes))
                    cached = self._cache.get(key)
                    if isinstance(cached, dict) and "summary" in cached and "diagnostics" in cached:
                        diag = dict(cached.get("diagnostics") or {})
                        diag.setdefault("cache", {})
                        diag["cache"] = {
                            "hit": True,
                            "fingerprint": fp,
                            "fingerprint_diag": fp_diag,
                        }
                        return ToolResult(summary=str(cached.get("summary") or ""), diagnostics=diag)
        sections, fetch_diag = await fetch_section_nodes(
            adapter=request.adapter,
            access_scope=request.access_scope,
            file_id=file_id,
            file_id_raw=file_id_raw,
            include_node_types=True,
        )
        tree, printed, orphaned = build_section_tree(sections)
        lines = self._format_tree(tree, max_depth=max_depth, max_nodes=max_nodes)
        truncated = len(lines) >= max_nodes
        summary = "toc.tree returned section tree:\n" + "\n".join(lines)
        if truncated:
            summary += "\n... (truncated)"
        diagnostics = {
            **fetch_diag,
            "tree": tree,
            "printed_nodes": printed,
            "orphaned_nodes": orphaned,
            "truncated": truncated,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "toc_source": "neo4j_section",
        }
        if not sections:
            reason = diagnostics.get("reason") or "no_sections_found"
            diagnostics["reason"] = reason
            if reason == "pageindex_disabled":
                return ToolResult(summary="toc.tree skipped: PageIndex disabled.", diagnostics=diagnostics)
            if reason == "cypher_unavailable":
                return ToolResult(summary="toc.tree skipped: Cypher unavailable.", diagnostics=diagnostics)
            if reason == "owner_scope_missing":
                return ToolResult(summary="toc.tree skipped: missing owner scope.", diagnostics=diagnostics)
            return ToolResult(summary="toc.tree returned no sections (PageIndex Section nodes missing).", diagnostics=diagnostics)

        if self._cache is not None and request.access_scope is not None:
            try:
                owner_scope_id = str(getattr(request.access_scope, "scope_id", "") or "")
            except Exception:
                owner_scope_id = ""
            if owner_scope_id:
                fp, fp_diag = await fetch_section_tree_fingerprint(
                    adapter=request.adapter,
                    access_scope=request.access_scope,
                    file_id=file_id,
                )
                if fp:
                    key = (owner_scope_id, file_id, fp, int(max_depth), int(max_nodes))
                    diagnostics.setdefault("cache", {})
                    diagnostics["cache"] = {"hit": False, "fingerprint": fp, "fingerprint_diag": fp_diag}
                    try:
                        self._cache.set(key, {"summary": summary, "diagnostics": diagnostics})
                    except Exception:
                        pass
        return ToolResult(summary=summary, diagnostics=diagnostics)

    @staticmethod
    def _format_tree(tree: Dict[str, Any], *, max_depth: int, max_nodes: int) -> List[str]:
        lines: List[str] = []

        def _walk(node: Dict[str, Any], depth: int) -> None:
            if len(lines) >= max_nodes:
                return
            if depth > max_depth:
                return
            children = node.get("children") if isinstance(node.get("children"), list) else []
            for child in children:
                if len(lines) >= max_nodes:
                    return
                if not isinstance(child, dict):
                    continue
                title = str(child.get("title") or "").strip()
                sid = str(child.get("section_id") or "").strip()
                page_start = child.get("page_start")
                page_end = child.get("page_end")
                page_hint = ""
                if isinstance(page_start, int) or isinstance(page_end, int):
                    page_hint = f" (p{page_start}-{page_end})"
                node_types = child.get("node_types") if isinstance(child.get("node_types"), dict) else {}
                node_tags = []
                for key in ("table", "image", "equation"):
                    count = node_types.get(key)
                    if isinstance(count, int) and count > 0:
                        node_tags.append(f"{key}={count}")
                node_hint = f" [{', '.join(node_tags)}]" if node_tags else ""
                label = title
                if sid:
                    label = f"{title} [section_id={sid}]"
                indent = "  " * max(0, depth - 1)
                lines.append(f"{indent}- {label}{page_hint}{node_hint}".rstrip())
                _walk(child, depth + 1)

        _walk(tree, 1)
        return lines
