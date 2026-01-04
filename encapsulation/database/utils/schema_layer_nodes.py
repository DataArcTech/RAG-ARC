"""Utilities for persisting schema-layer nodes derived from chunk metadata.

Currently this consumes the per-chunk mindmap (TSV → nodes[]) extracted by HippoRAG2 and
turns it into:
- schema nodes (global per owner, merged by (layer, normalized_text))
- per-chunk links from Chunk → SchemaNode with mindmap level metadata
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing


_LAYER_TAG_RE = re.compile(r"^\[(?P<layer>[A-Za-z_][A-Za-z0-9_]*)\]\s*(?P<text>.*)$")


def _parse_layer_tag(content: str) -> tuple[str | None, str]:
    raw = str(content or "").strip()
    if not raw:
        return None, ""
    match = _LAYER_TAG_RE.match(raw)
    if not match:
        return None, raw
    layer = (match.group("layer") or "").strip().lower()
    text = (match.group("text") or "").strip()
    return (layer or None), text


def build_schema_layer_payload(
    *,
    mindmap_nodes: Iterable[Dict[str, Any]] | None,
    chunk_id: str,
    owner_id: Optional[str],
    db_owner_id: str,
    max_nodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (schema_nodes, schema_links) to be batch-upserted into Neo4j.

    - schema_nodes are merged by a stable schema_id (owner-scoped).
    - schema_links are chunk-scoped occurrences, carrying the mindmap 'level' and 'layer'.
    """

    chunk_id = str(chunk_id or "").strip()
    if not chunk_id:
        return [], []

    limit = max(0, int(max_nodes))
    if limit == 0:
        return [], []

    nodes = list(mindmap_nodes or [])
    schema_nodes_by_id: dict[str, dict[str, Any]] = {}
    schema_links: list[dict[str, Any]] = []

    for item in nodes[:limit]:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level") or "").strip()
        raw_content = str(item.get("content") or "").strip()
        layer, text = _parse_layer_tag(raw_content)
        cleaned = text.strip()
        if not cleaned:
            continue

        normalized = text_processing(cleaned)
        if not normalized:
            continue
        layer_token = layer or "unknown"
        schema_key = f"{layer_token}|{normalized}"
        schema_id = compute_mdhash_id(schema_key, prefix="schema-", owner_id=owner_id)
        schema_nodes_by_id.setdefault(
            schema_id,
            {
                "schema_id": schema_id,
                "layer": layer_token,
                "text": cleaned,
                "text_normalized": normalized,
                "owner_id": db_owner_id,
            },
        )
        schema_links.append(
            {
                "chunk_id": chunk_id,
                "schema_id": schema_id,
                "level": level,
                "layer": layer_token,
                "owner_id": db_owner_id,
            }
        )

    return list(schema_nodes_by_id.values()), schema_links

