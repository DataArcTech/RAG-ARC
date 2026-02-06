"""Section tree helpers for PageIndex-backed navigation."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import pageindex as pageindex_cfg
from config.core.deepsearch import tool_defaults
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher
from core.deepsearch.utils.ids import normalize_uuid
from core.deepsearch.utils.node_types import normalize_node_type
from core.utils.json_extract import safe_json_loads


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


@dataclass
class SectionNode:
    section_id: str
    path: str
    title: str
    level: Optional[int]
    page_start: Optional[int]
    page_end: Optional[int]
    parent_id: Optional[str]
    node_types: Dict[str, int] = field(default_factory=dict)


async def fetch_section_nodes(
    *,
    adapter: Any,
    access_scope: Any,
    file_id: str,
    file_id_raw: str | None = None,
    include_node_types: bool = False,
) -> tuple[List[SectionNode], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"file_id": file_id}
    if file_id_raw and file_id_raw != file_id:
        diagnostics["file_id_raw"] = file_id_raw
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return [], diagnostics
    if adapter is None or not adapter_supports_cypher(adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return [], diagnostics
    if access_scope is None or not getattr(access_scope, "scope_id", None):
        diagnostics["reason"] = "owner_scope_missing"
        return [], diagnostics

    cypher = (
        "MATCH (s:Section)\n"
        "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
        "RETURN s.section_id AS section_id,\n"
        "       s.section_path AS section_path,\n"
        "       s.section_title AS section_title,\n"
        "       s.section_level AS section_level,\n"
        "       s.page_start AS page_start,\n"
        "       s.page_end AS page_end,\n"
        "       s.section_parent_id AS section_parent_id\n"
    )
    params = {"file_id": file_id}
    async with adapter_locked(adapter):
        rows = await adapter.acypher(cypher, params, access_scope=access_scope)

    sections: List[SectionNode] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        section_id = str(row.get("section_id") or "").strip()
        section_path = str(row.get("section_path") or "").strip()
        title = str(row.get("section_title") or "").strip()
        if not section_id or not section_path:
            continue
        sections.append(
            SectionNode(
                section_id=section_id,
                path=section_path,
                title=title,
                level=_coerce_int(row.get("section_level")),
                page_start=_coerce_int(row.get("page_start")),
                page_end=_coerce_int(row.get("page_end")),
                parent_id=str(row.get("section_parent_id") or "").strip() or None,
            )
        )

    sections.sort(key=lambda s: (s.page_start if s.page_start is not None else 1_000_000, s.path))
    diagnostics["rows_scanned"] = len(rows or [])
    diagnostics["sections"] = len(sections)

    if include_node_types and sections:
        node_types, node_diag = await fetch_section_node_types(
            adapter=adapter,
            access_scope=access_scope,
            file_id=file_id,
            section_ids=[s.section_id for s in sections],
        )
        diagnostics["node_type_counts"] = node_diag
        for node in sections:
            node.node_types = node_types.get(node.section_id, {})

    return sections, diagnostics


async def fetch_section_tree_fingerprint(
    *,
    adapter: Any,
    access_scope: Any,
    file_id: str,
) -> tuple[str | None, Dict[str, Any]]:
    """Return a cheap fingerprint for a file's Section tree (for cache busting).

    We intentionally avoid reading all Section rows here; this query returns a single row.
    """

    diagnostics: Dict[str, Any] = {"file_id": file_id}
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return None, diagnostics
    if adapter is None or not adapter_supports_cypher(adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return None, diagnostics
    if access_scope is None or not getattr(access_scope, "scope_id", None):
        diagnostics["reason"] = "owner_scope_missing"
        return None, diagnostics

    cypher = (
        "MATCH (s:Section)\n"
        "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
        "RETURN count(s) AS section_count, max(s.updated_at) AS max_updated_at\n"
    )
    params = {"file_id": file_id}
    async with adapter_locked(adapter):
        rows = await adapter.acypher(cypher, params, access_scope=access_scope)
    row = rows[0] if isinstance(rows, list) and rows else {}
    if not isinstance(row, dict):
        row = {}
    try:
        count = int(row.get("section_count") or 0)
    except Exception:
        count = 0
    updated_at = row.get("max_updated_at")
    updated_str = str(updated_at) if updated_at is not None else ""
    fingerprint = f"sections:{count}|updated:{updated_str}"
    diagnostics.update({"section_count": count, "max_updated_at": updated_str})
    return fingerprint, diagnostics


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        parsed = safe_json_loads(raw, expected="dict")
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


async def fetch_section_node_types(
    *,
    adapter: Any,
    access_scope: Any,
    file_id: str,
    section_ids: List[str],
) -> tuple[Dict[str, Dict[str, int]], Dict[str, Any]]:
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

    max_chunks = int(tool_defaults.SECTION_NODE_TYPE_MAX_CHUNKS)
    max_chunks = max(1, max_chunks)
    cypher = (
        "UNWIND $section_ids AS sid\n"
        "MATCH (s:Section {section_id: sid})-[:HAS_CHUNK]->(c:Chunk)\n"
        "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
        "RETURN sid AS section_id, c.metadata AS metadata\n"
        "LIMIT $limit\n"
    )
    params = {"section_ids": section_ids, "file_id": file_id, "limit": max_chunks}
    async with adapter_locked(adapter):
        rows = await adapter.acypher(cypher, params, access_scope=access_scope)

    counts: Dict[str, Dict[str, int]] = {}
    total = 0
    used = 0
    per_section_used: Dict[str, int] = {}
    max_per_section = int(tool_defaults.SECTION_NODE_TYPE_MAX_PER_SECTION)
    max_per_section = max(1, max_per_section)
    for row in rows or []:
        total += 1
        if not isinstance(row, dict):
            continue
        sid = str(row.get("section_id") or "").strip()
        if not sid:
            continue
        used_for_section = per_section_used.get(sid, 0)
        if used_for_section >= max_per_section:
            continue
        meta = _parse_metadata(row.get("metadata"))
        node_type = normalize_node_type(meta.get("semantic_unit_type"))
        bucket = counts.setdefault(sid, {})
        bucket[node_type] = bucket.get(node_type, 0) + 1
        per_section_used[sid] = used_for_section + 1
        used += 1

    diagnostics.update(
        {
            "rows_scanned": total,
            "rows_used": used,
            "max_chunks": max_chunks,
            "max_per_section": max_per_section,
            "sections_counted": len(counts),
        }
    )
    if total >= max_chunks:
        diagnostics["truncated"] = True
    return counts, diagnostics


def build_section_tree(sections: List[SectionNode]) -> Tuple[Dict[str, Any], int, int]:
    root: Dict[str, Any] = {"title": None, "children": []}
    nodes: Dict[str, Dict[str, Any]] = {}
    printed = 0
    orphaned = 0
    for item in sections:
        nodes[item.section_id] = {
            "title": item.title or item.path,
            "children": [],
            "section_id": item.section_id,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "level": item.level,
            "parent_id": item.parent_id,
            "section_path": item.path,
            "node_types": dict(item.node_types or {}),
        }
        printed += 1

    for item in sections:
        node = nodes.get(item.section_id)
        if node is None:
            continue
        parent_id = item.parent_id
        parent = nodes.get(parent_id) if parent_id else None
        if parent is None:
            orphaned += 1
            root["children"].append(node)
        else:
            parent["children"].append(node)

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
    return root, printed, orphaned


def normalize_file_id(file_id_raw: str | None) -> tuple[str | None, str | None]:
    token = str(file_id_raw or "").strip()
    if not token:
        return None, None
    normalized = normalize_uuid(token)
    if not normalized:
        return None, token
    return normalized, token
