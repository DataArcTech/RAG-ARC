"""toc.tree tool: list a file's section tree (PageIndex navigation).

This tool is intentionally "structure first": it does not do semantic retrieval.
It scans chunk metadata for `section_path/section_id/section_level/page_*` and
reconstructs a readable ToC tree for the LLM to pick a section before reading.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.core.deepsearch import tool_defaults
from config import pageindex as pageindex_cfg
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER

_PATH_DELIM = " > "


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    # Neo4j drivers may return maps or JSON-encoded strings depending on adapter/storage.
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


@dataclass
class _SectionInfo:
    section_id: str
    path: str
    level: Optional[int]
    page_start: Optional[int]
    page_end: Optional[int]


class TocTreeTool(GraphTool):
    descriptor = ToolDescriptor(
        name="toc.tree",
        channel="graph",
        description="List a file's section tree (ToC) reconstructed from PageIndex-enriched chunk metadata.",
        speed="fast",
        cost="low",
        strategy_tags=("toc", "pageindex", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.toc_tree",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "Target file_id (required)."},
                "max_depth": {"type": "integer", "minimum": 1, "description": "Max ToC depth to print."},
                "max_nodes": {"type": "integer", "minimum": 1, "description": "Max nodes to print before truncating."},
            },
            required_extra_fields=("file_id",),
        ),
        example_args={
            "question": "Show ToC for the manual",
            "plan_step": "plan_01",
            "extra": {"file_id": "<file_id>", "max_depth": 4},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        if not pageindex_cfg.pageindex_enabled():
            return ToolResult(summary="toc.tree skipped: PageIndex disabled.", diagnostics={"reason": "pageindex_disabled"})
        if request.adapter is None or not adapter_supports_cypher(request.adapter):
            return ToolResult(summary="toc.tree skipped: Cypher unavailable.", diagnostics={"reason": "cypher_unavailable"})
        if request.access_scope is None or not getattr(request.access_scope, "scope_id", None):
            return ToolResult(summary="toc.tree skipped: missing owner scope.", diagnostics={"reason": "owner_scope_missing"})

        extra = request.extra or {}
        file_id = str(extra.get("file_id") or "").strip()
        if not file_id:
            return ToolResult(summary="toc.tree skipped: missing file_id.", diagnostics={"reason": "missing_file_id"})

        max_depth = _coerce_int(extra.get("max_depth")) or int(tool_defaults.TOC_TREE_DEFAULT_MAX_DEPTH)
        max_nodes = _coerce_int(extra.get("max_nodes")) or int(tool_defaults.TOC_TREE_MAX_NODES)
        max_depth = max(1, max_depth)
        max_nodes = max(1, max_nodes)
        scan_limit = int(tool_defaults.TOC_TREE_MAX_CHUNKS_SCANNED)

        cypher = (
            "MATCH (c:Chunk)\n"
            "WHERE c.source_file_id = $file_id AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
            "RETURN c.metadata AS metadata\n"
            "LIMIT $limit\n"
        )
        params = {"file_id": file_id, "limit": max(1, scan_limit)}
        async with adapter_locked(request.adapter):
            rows = await request.adapter.acypher(cypher, params, access_scope=request.access_scope)

        unique: Dict[str, _SectionInfo] = {}
        missing_meta = 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            meta = _parse_metadata(row.get("metadata"))
            if not meta:
                missing_meta += 1
                continue
            section_id = str(meta.get("section_id") or "").strip()
            section_path = str(meta.get("section_path") or "").strip()
            if not section_id or not section_path:
                continue
            info = unique.get(section_id)
            page_start = _coerce_int(meta.get("page_start"))
            page_end = _coerce_int(meta.get("page_end"))
            level = _coerce_int(meta.get("section_level"))
            if info is None:
                unique[section_id] = _SectionInfo(
                    section_id=section_id,
                    path=section_path,
                    level=level,
                    page_start=page_start,
                    page_end=page_end,
                )
            else:
                # Merge page ranges conservatively (best-effort).
                if page_start is not None:
                    info.page_start = page_start if info.page_start is None else min(info.page_start, page_start)
                if page_end is not None:
                    info.page_end = page_end if info.page_end is None else max(info.page_end, page_end)

        sections = list(unique.values())
        sections.sort(key=lambda s: (s.page_start if s.page_start is not None else 1_000_000, s.path))

        tree, printed = self._build_tree(sections)
        lines = self._format_tree(tree, max_depth=max_depth, max_nodes=max_nodes)
        truncated = len(lines) >= max_nodes
        summary = "toc.tree returned section tree:\n" + "\n".join(lines)
        if truncated:
            summary += "\n... (truncated)"
        diagnostics = {
            "file_id": file_id,
            "sections": len(sections),
            "scan_limit": scan_limit,
            "rows_scanned": len(rows or []),
            "missing_metadata_rows": missing_meta,
            "tree": tree,
            "printed_nodes": printed,
            "truncated": truncated,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        }
        if not sections:
            diagnostics["reason"] = "no_sections_found"
            return ToolResult(summary="toc.tree returned no sections (missing PageIndex section metadata).", diagnostics=diagnostics)
        return ToolResult(summary=summary, diagnostics=diagnostics)

    @staticmethod
    def _build_tree(sections: List[_SectionInfo]) -> Tuple[Dict[str, Any], int]:
        root: Dict[str, Any] = {"title": None, "children": []}
        nodes: Dict[Tuple[str, ...], Dict[str, Any]] = {(): root}
        printed = 0
        for item in sections:
            parts = [p.strip() for p in item.path.split(_PATH_DELIM) if p.strip()]
            if not parts:
                continue
            for depth in range(1, len(parts) + 1):
                key = tuple(parts[:depth])
                parent_key = tuple(parts[: depth - 1])
                if key not in nodes:
                    node = {"title": parts[depth - 1], "children": []}
                    nodes[key] = node
                    nodes[parent_key]["children"].append(node)
                if depth == len(parts):
                    nodes[key]["section_id"] = item.section_id
                    nodes[key]["page_start"] = item.page_start
                    nodes[key]["page_end"] = item.page_end
                    nodes[key]["level"] = item.level
            printed += 1

        def _sort(node: Dict[str, Any]) -> None:
            children = node.get("children") if isinstance(node.get("children"), list) else []
            children.sort(
                key=lambda c: (
                    c.get("page_start") if isinstance(c, dict) and isinstance(c.get("page_start"), int) else 1_000_000,
                    str(c.get("title") or "") if isinstance(c, dict) else "",
                )
            )
            for child in children:
                if isinstance(child, dict):
                    _sort(child)

        _sort(root)
        return root, printed

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
                label = title
                if sid:
                    label = f"{title} [section_id={sid}]"
                indent = "  " * max(0, depth - 1)
                lines.append(f"{indent}- {label}{page_hint}".rstrip())
                _walk(child, depth + 1)

        _walk(tree, 1)
        return lines
