"""Tree navigation tools for PageIndex agentic retrieval."""
from typing import Any, Dict, List, Optional

from config import pageindex as pageindex_cfg
from config.core.deepsearch import tool_defaults
from core.deepsearch.utils.section_tree import fetch_section_nodes, normalize_file_id
from core.deepsearch.utils.tree_nodes import (
    TreeNode,
    fetch_section_tree_stats,
    fetch_tree_children,
    fetch_tree_node,
    fetch_tree_nodes_for_section,
)

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, SCOPE_FILE, SCOPE_OWNER


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _page_span(node: TreeNode) -> Optional[int]:
    if node.page_start is None or node.page_end is None:
        return None
    return max(0, node.page_end - node.page_start + 1)


def _page_hint(page_start: Optional[int], page_end: Optional[int]) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_end is None:
        return f" (p{page_start})"
    return f" (p{page_start}-{page_end})"


def _format_node_line(node: TreeNode) -> str:
    title = (node.summary or "").strip() or node.node_type
    node_id = node.node_id
    page_hint = _page_hint(node.page_start, node.page_end)
    return f"- {node.node_type}: {title} [node_id={node_id}]{page_hint}".rstrip()


class TreeRootTool(GraphTool):
    descriptor = ToolDescriptor(
        name="tree.root",
        channel="graph",
        description="Show top-level sections and TreeNode stats for a file (display-only).",
        speed="fast",
        cost="low",
        strategy_tags=("tree", "pageindex", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.tree.root",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "Target file_id (required)."},
                "max_roots": {"type": "integer", "minimum": 1, "description": "Max root sections to print."},
            },
            required_extra_fields=("file_id",),
        ),
        example_args={
            "question": "Show root of the document",
            "plan_step": "plan_01",
            "extra": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "max_roots": 12},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        file_id, file_id_raw = normalize_file_id(extra.get("file_id"))
        if not file_id:
            reason = "missing_file_id" if not file_id_raw else "invalid_file_id_format"
            return ToolResult(
                summary="tree.root skipped: invalid/missing file_id (expected UUID; use search.file).",
                diagnostics={"reason": reason, "file_id_raw": file_id_raw or None},
            )

        max_roots = _coerce_int(extra.get("max_roots")) or pageindex_cfg.toc_check_page_num()
        max_roots = max(1, max_roots)
        sections, fetch_diag = await fetch_section_nodes(
            adapter=request.adapter,
            access_scope=request.access_scope,
            file_id=file_id,
            file_id_raw=file_id_raw,
            include_node_types=True,
        )
        root_sections = [s for s in sections if not s.parent_id]
        stats, stats_diag = await fetch_section_tree_stats(
            adapter=request.adapter,
            access_scope=request.access_scope,
            file_id=file_id,
            section_ids=[s.section_id for s in root_sections],
        )

        max_pages = pageindex_cfg.max_page_num_each_node()
        max_tokens = pageindex_cfg.max_token_num_each_node()

        lines: List[str] = []
        diagnostics = {**fetch_diag, "root_sections": len(root_sections), "stats": stats_diag}
        for item in root_sections[:max_roots]:
            node_types = item.node_types or {}
            node_tags = []
            for key in ("table", "image", "equation"):
                count = node_types.get(key)
                if isinstance(count, int) and count > 0:
                    node_tags.append(f"{key}={count}")
            tag = f" [{', '.join(node_tags)}]" if node_tags else ""
            page_hint = _page_hint(item.page_start, item.page_end)
            stats_row = stats.get(item.section_id, {})
            token_count = int(stats_row.get("token_count") or 0)
            node_count = int(stats_row.get("node_count") or 0)
            span = None
            if item.page_start is not None and item.page_end is not None:
                span = max(0, item.page_end - item.page_start + 1)
            oversized = bool(span and span > max_pages and token_count >= max_tokens)
            oversized_tag = " [oversized]" if oversized else ""
            line = (
                f"- {item.title or item.path} [section_id={item.section_id}]"
                f"{page_hint}{tag} nodes={node_count} tokens≈{token_count}{oversized_tag}"
            )
            lines.append(line)

        summary = "tree.root returned top-level sections:\n" + "\n".join(lines)
        diagnostics.update(
            {
                "max_roots": max_roots,
                "max_page_num_each_node": max_pages,
                "max_token_num_each_node": max_tokens,
            }
        )
        if not root_sections:
            reason = diagnostics.get("reason") or "no_sections_found"
            diagnostics["reason"] = reason
            return ToolResult(summary="tree.root returned no sections.", diagnostics=diagnostics)
        return ToolResult(summary=summary, diagnostics=diagnostics)


class TreeChildrenTool(GraphTool):
    descriptor = ToolDescriptor(
        name="tree.children",
        channel="graph",
        description="List children TreeNodes under a section or a TreeNode.",
        speed="fast",
        cost="low",
        strategy_tags=("tree", "pageindex", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.tree.children",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "Target file_id (required for section lookup)."},
                "section_id": {"type": "string", "description": "Section id to list child TreeNodes."},
                "node_id": {"type": "string", "description": "TreeNode id to list child TreeNodes."},
                "max_nodes": {"type": "integer", "minimum": 1, "description": "Max nodes to return."},
            },
            required_extra_fields=(),
        ),
        example_args={
            "question": "List nodes in section",
            "plan_step": "plan_01",
            "extra": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "section_id": "REPLACE_WITH_SECTION_ID", "max_nodes": 50},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        node_id = str(extra.get("node_id") or "").strip()
        section_id = str(extra.get("section_id") or "").strip()
        file_id, file_id_raw = normalize_file_id(extra.get("file_id"))
        max_nodes = _coerce_int(extra.get("max_nodes")) or int(tool_defaults.TOC_TREE_MAX_NODES)
        max_nodes = max(1, max_nodes)

        if node_id:
            nodes, diagnostics = await fetch_tree_children(
                adapter=request.adapter,
                access_scope=request.access_scope,
                node_id=node_id,
                max_nodes=max_nodes,
            )
        else:
            if not file_id:
                reason = "missing_file_id" if not file_id_raw else "invalid_file_id_format"
                return ToolResult(
                    summary="tree.children skipped: missing/invalid file_id (expected UUID; use search.file).",
                    diagnostics={"reason": reason, "file_id_raw": file_id_raw or None},
                )
            nodes, diagnostics = await fetch_tree_nodes_for_section(
                adapter=request.adapter,
                access_scope=request.access_scope,
                file_id=file_id,
                section_id=section_id,
                max_nodes=max_nodes,
            )

        lines = [_format_node_line(node) for node in nodes]
        summary = "tree.children returned TreeNodes:\n" + "\n".join(lines)
        if not nodes:
            reason = diagnostics.get("reason") or "no_nodes_found"
            diagnostics["reason"] = reason
            return ToolResult(summary="tree.children returned no nodes.", diagnostics=diagnostics)

        diagnostics["suggested_reads"] = [
            {
                "file_id": node.file_id,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "node_id": node.node_id,
            }
            for node in nodes
            if node.file_id and node.page_start is not None and node.page_end is not None
        ]
        return ToolResult(summary=summary, diagnostics=diagnostics)


class TreeNodeTool(GraphTool):
    descriptor = ToolDescriptor(
        name="tree.node",
        channel="graph",
        description="Show metadata for a TreeNode (display-only; use read.pages for evidence).",
        speed="fast",
        cost="low",
        strategy_tags=("tree", "pageindex", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.tree.node",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "node_id": {"type": "string", "description": "TreeNode id (required)."},
            },
            required_extra_fields=("node_id",),
        ),
        example_args={
            "question": "Inspect this node",
            "plan_step": "plan_02",
            "extra": {"node_id": "NODE_ID_FROM_TREE_ROOT_OR_CHILDREN"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        node_id = str(extra.get("node_id") or "").strip()
        node, diagnostics = await fetch_tree_node(
            adapter=request.adapter,
            access_scope=request.access_scope,
            node_id=node_id,
        )
        if not node:
            reason = diagnostics.get("reason") or "node_not_found"
            diagnostics["reason"] = reason
            return ToolResult(summary="tree.node returned no node.", diagnostics=diagnostics)

        summary = (
            "tree.node:\n"
            f"- node_id={node.node_id}\n"
            f"- type={node.node_type}\n"
            f"- section_id={node.section_id or ''}\n"
            f"- page_range={node.page_start}-{node.page_end}\n"
            f"- summary={node.summary or ''}"
        )
        diagnostics["suggested_read"] = {
            "file_id": node.file_id,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "node_id": node.node_id,
        }
        return ToolResult(summary=summary, diagnostics=diagnostics)


class TreeOpenTool(GraphTool):
    descriptor = ToolDescriptor(
        name="tree.open",
        channel="graph",
        description="Open a TreeNode summary + resource hints (display-only; use read.pages for evidence).",
        speed="fast",
        cost="low",
        strategy_tags=("tree", "pageindex", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.tree.open",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "node_id": {"type": "string", "description": "TreeNode id (preferred)."},
                "file_id": {"type": "string", "description": "Target file_id (required when using section_id)."},
                "section_id": {"type": "string", "description": "Section id fallback (use when node_id unknown)."},
            },
            required_extra_fields=(),
        ),
        example_args={
            "question": "Open a TreeNode to see summary and resource hints",
            "plan_step": "plan_02",
            "extra": {"node_id": "NODE_ID_FROM_TREE_ROOT_OR_CHILDREN"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        node_id = str(extra.get("node_id") or "").strip()
        section_id = str(extra.get("section_id") or "").strip()
        file_id, file_id_raw = normalize_file_id(extra.get("file_id")) if section_id else (None, None)
        diagnostics: Dict[str, Any] = {}

        node = None
        if node_id:
            node, diagnostics = await fetch_tree_node(
                adapter=request.adapter,
                access_scope=request.access_scope,
                node_id=node_id,
            )
        elif section_id:
            if not file_id:
                reason = "missing_file_id" if not file_id_raw else "invalid_file_id_format"
                return ToolResult(
                    summary="tree.open skipped: missing/invalid file_id for section_id fallback.",
                    diagnostics={"reason": reason, "file_id_raw": file_id_raw or None, "section_id": section_id},
                )
            nodes, diagnostics = await fetch_tree_nodes_for_section(
                adapter=request.adapter,
                access_scope=request.access_scope,
                file_id=file_id,
                section_id=section_id,
                max_nodes=max(1, int(tool_defaults.TOC_TREE_MAX_NODES)),
            )
            node = nodes[0] if nodes else None
            diagnostics["fallback"] = "section_id"
            diagnostics["section_id"] = section_id
        else:
            return ToolResult(
                summary="tree.open skipped: missing node_id (or section_id).",
                diagnostics={"reason": "missing_node_id"},
            )

        if not node:
            reason = diagnostics.get("reason") or "node_not_found"
            diagnostics["reason"] = reason
            return ToolResult(summary="tree.open returned no node.", diagnostics=diagnostics)

        summary_lines = [
            "tree.open:",
            f"- node_id={node.node_id}",
            f"- section_id={node.section_id or ''}",
            f"- type={node.node_type}",
            f"- page_range={node.page_start}-{node.page_end}",
        ]
        if node.summary:
            summary_lines.append(f"- summary={node.summary}")
        if node.resource_urls:
            summary_lines.append(f"- resource_urls={len(node.resource_urls)}")
        summary = "\n".join(summary_lines)
        diagnostics["resource_urls"] = node.resource_urls
        diagnostics["suggested_read"] = {
            "file_id": node.file_id,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "node_id": node.node_id,
        }
        return ToolResult(summary=summary, diagnostics=diagnostics)


__all__ = [
    "TreeRootTool",
    "TreeChildrenTool",
    "TreeNodeTool",
    "TreeOpenTool",
]
