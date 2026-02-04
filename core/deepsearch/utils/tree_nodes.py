"""TreeNode helpers for PageIndex agentic navigation."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import pageindex as pageindex_cfg
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TreeNode:
    node_id: str
    node_type: str
    page_start: Optional[int]
    page_end: Optional[int]
    summary: Optional[str]
    section_id: Optional[str]
    section_path: Optional[str]
    file_id: Optional[str]
    resource_urls: List[str]
    token_count: Optional[int]


async def fetch_tree_nodes_for_section(
    *,
    adapter: Any,
    access_scope: Any,
    file_id: str,
    section_id: str,
    max_nodes: int,
) -> Tuple[List[TreeNode], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"file_id": file_id, "section_id": section_id}
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return [], diagnostics
    if adapter is None or not adapter_supports_cypher(adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return [], diagnostics
    if access_scope is None or not getattr(access_scope, "scope_id", None):
        diagnostics["reason"] = "owner_scope_missing"
        return [], diagnostics
    if not section_id:
        diagnostics["reason"] = "missing_section_id"
        return [], diagnostics

    limit = max(1, int(max_nodes))
    cypher = (
        "MATCH (s:Section {section_id: $section_id})-[:HAS_CHILD]->(t:TreeNode)\n"
        "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
        "RETURN t.node_id AS node_id,\n"
        "       t.node_type AS node_type,\n"
        "       t.page_start AS page_start,\n"
        "       t.page_end AS page_end,\n"
        "       t.summary AS summary,\n"
        "       t.section_id AS section_id,\n"
        "       t.section_path AS section_path,\n"
        "       t.source_file_id AS file_id,\n"
        "       t.resource_urls AS resource_urls,\n"
        "       t.resource_paths AS resource_paths,\n"
        "       t.token_count AS token_count\n"
        "ORDER BY t.page_start, t.node_id\n"
        "LIMIT $limit\n"
    )
    params = {"file_id": file_id, "section_id": section_id, "limit": limit}
    async with adapter_locked(adapter):
        rows = await adapter.acypher(cypher, params, access_scope=access_scope)

    nodes: List[TreeNode] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("node_id") or "").strip()
        if not node_id:
            continue
        resource_urls = row.get("resource_urls") if isinstance(row.get("resource_urls"), list) else []
        if not resource_urls:
            resource_urls = row.get("resource_paths") if isinstance(row.get("resource_paths"), list) else []
        nodes.append(
            TreeNode(
                node_id=node_id,
                node_type=str(row.get("node_type") or "page").strip(),
                page_start=_coerce_int(row.get("page_start")),
                page_end=_coerce_int(row.get("page_end")),
                summary=str(row.get("summary") or "").strip() or None,
                section_id=str(row.get("section_id") or "").strip() or None,
                section_path=str(row.get("section_path") or "").strip() or None,
                file_id=str(row.get("file_id") or "").strip() or None,
                resource_urls=[str(u).strip() for u in resource_urls if str(u or "").strip()],
                token_count=_coerce_int(row.get("token_count")),
            )
        )

    diagnostics["nodes"] = len(nodes)
    if len(nodes) >= limit:
        diagnostics["truncated"] = True
    return nodes, diagnostics


async def fetch_tree_children(
    *,
    adapter: Any,
    access_scope: Any,
    node_id: str,
    max_nodes: int,
) -> Tuple[List[TreeNode], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"node_id": node_id}
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return [], diagnostics
    if adapter is None or not adapter_supports_cypher(adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return [], diagnostics
    if access_scope is None or not getattr(access_scope, "scope_id", None):
        diagnostics["reason"] = "owner_scope_missing"
        return [], diagnostics
    if not node_id:
        diagnostics["reason"] = "missing_node_id"
        return [], diagnostics

    limit = max(1, int(max_nodes))
    cypher = (
        "MATCH (p:TreeNode {node_id: $node_id, owner_id: $owner_id})-[:HAS_CHILD]->(t:TreeNode)\n"
        "RETURN t.node_id AS node_id,\n"
        "       t.node_type AS node_type,\n"
        "       t.page_start AS page_start,\n"
        "       t.page_end AS page_end,\n"
        "       t.summary AS summary,\n"
        "       t.section_id AS section_id,\n"
        "       t.section_path AS section_path,\n"
        "       t.source_file_id AS file_id,\n"
        "       t.resource_urls AS resource_urls,\n"
        "       t.resource_paths AS resource_paths,\n"
        "       t.token_count AS token_count\n"
        "ORDER BY t.page_start, t.node_id\n"
        "LIMIT $limit\n"
    )
    params = {"node_id": node_id, "limit": limit}
    async with adapter_locked(adapter):
        rows = await adapter.acypher(cypher, params, access_scope=access_scope)

    nodes: List[TreeNode] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        child_id = str(row.get("node_id") or "").strip()
        if not child_id:
            continue
        resource_urls = row.get("resource_urls") if isinstance(row.get("resource_urls"), list) else []
        if not resource_urls:
            resource_urls = row.get("resource_paths") if isinstance(row.get("resource_paths"), list) else []
        nodes.append(
            TreeNode(
                node_id=child_id,
                node_type=str(row.get("node_type") or "page").strip(),
                page_start=_coerce_int(row.get("page_start")),
                page_end=_coerce_int(row.get("page_end")),
                summary=str(row.get("summary") or "").strip() or None,
                section_id=str(row.get("section_id") or "").strip() or None,
                section_path=str(row.get("section_path") or "").strip() or None,
                file_id=str(row.get("file_id") or "").strip() or None,
                resource_urls=[str(u).strip() for u in resource_urls if str(u or "").strip()],
                token_count=_coerce_int(row.get("token_count")),
            )
        )

    diagnostics["nodes"] = len(nodes)
    if len(nodes) >= limit:
        diagnostics["truncated"] = True
    return nodes, diagnostics


async def fetch_tree_node(
    *,
    adapter: Any,
    access_scope: Any,
    node_id: str,
) -> Tuple[Optional[TreeNode], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"node_id": node_id}
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return None, diagnostics
    if adapter is None or not adapter_supports_cypher(adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return None, diagnostics
    if access_scope is None or not getattr(access_scope, "scope_id", None):
        diagnostics["reason"] = "owner_scope_missing"
        return None, diagnostics
    if not node_id:
        diagnostics["reason"] = "missing_node_id"
        return None, diagnostics

    cypher = (
        "MATCH (t:TreeNode {node_id: $node_id, owner_id: $owner_id})\n"
        "RETURN t.node_id AS node_id,\n"
        "       t.node_type AS node_type,\n"
        "       t.page_start AS page_start,\n"
        "       t.page_end AS page_end,\n"
        "       t.summary AS summary,\n"
        "       t.section_id AS section_id,\n"
        "       t.section_path AS section_path,\n"
        "       t.source_file_id AS file_id,\n"
        "       t.resource_urls AS resource_urls,\n"
        "       t.resource_paths AS resource_paths,\n"
        "       t.token_count AS token_count\n"
    )
    params = {"node_id": node_id}
    async with adapter_locked(adapter):
        rows = await adapter.acypher(cypher, params, access_scope=access_scope)
    row = rows[0] if rows else None
    if not isinstance(row, dict):
        diagnostics["reason"] = "node_not_found"
        return None, diagnostics

    resource_urls = row.get("resource_urls") if isinstance(row.get("resource_urls"), list) else []
    if not resource_urls:
        resource_urls = row.get("resource_paths") if isinstance(row.get("resource_paths"), list) else []
    node = TreeNode(
        node_id=str(row.get("node_id") or "").strip(),
        node_type=str(row.get("node_type") or "page").strip(),
        page_start=_coerce_int(row.get("page_start")),
        page_end=_coerce_int(row.get("page_end")),
        summary=str(row.get("summary") or "").strip() or None,
        section_id=str(row.get("section_id") or "").strip() or None,
        section_path=str(row.get("section_path") or "").strip() or None,
        file_id=str(row.get("file_id") or "").strip() or None,
        resource_urls=[str(u).strip() for u in resource_urls if str(u or "").strip()],
        token_count=_coerce_int(row.get("token_count")),
    )
    diagnostics["found"] = True
    return node, diagnostics


async def fetch_section_tree_stats(
    *,
    adapter: Any,
    access_scope: Any,
    file_id: str,
    section_ids: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"file_id": file_id}
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return {}, diagnostics
    if adapter is None or not adapter_supports_cypher(adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return {}, diagnostics
    if access_scope is None or not getattr(access_scope, "scope_id", None):
        diagnostics["reason"] = "owner_scope_missing"
        return {}, diagnostics

    section_ids = [str(sid).strip() for sid in section_ids if str(sid or "").strip()]
    if not section_ids:
        diagnostics["reason"] = "empty_section_ids"
        return {}, diagnostics

    cypher = (
        "UNWIND $section_ids AS sid\n"
        "MATCH (s:Section {section_id: sid})-[:HAS_CHILD]->(t:TreeNode)\n"
        "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
        "RETURN sid AS section_id,\n"
        "       count(t) AS node_count,\n"
        "       sum(COALESCE(t.token_count, 0)) AS token_count\n"
    )
    params = {"section_ids": section_ids, "file_id": file_id}
    async with adapter_locked(adapter):
        rows = await adapter.acypher(cypher, params, access_scope=access_scope)

    stats: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("section_id") or "").strip()
        if not sid:
            continue
        stats[sid] = {
            "node_count": int(row.get("node_count") or 0),
            "token_count": int(row.get("token_count") or 0),
        }
    diagnostics["sections"] = len(stats)
    return stats, diagnostics


__all__ = [
    "TreeNode",
    "fetch_tree_nodes_for_section",
    "fetch_tree_children",
    "fetch_tree_node",
    "fetch_section_tree_stats",
]
