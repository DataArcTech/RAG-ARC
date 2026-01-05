"""Utilities for persisting SDF schema fragments into Neo4j.

Input contract:
- `sdf` is expected to be a dict compatible with `core.knowledge_graph.sdf.SdfSchema`.
- Event IDs are expected to be stable strings (we treat them as opaque).
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from encapsulation.database.utils.pruned_hipporag_utils import text_processing


def build_sdf_schema_payload(
    *,
    sdf: Dict[str, Any] | None,
    chunk_id: str,
    db_owner_id: str,
    max_events: int,
    max_relations: int,
    max_source_chunks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (event_nodes, has_subevent_edges, before_edges, chunk_event_links)."""
    chunk_id = str(chunk_id or "").strip()
    if not chunk_id:
        return [], [], [], []
    if not isinstance(sdf, dict):
        return [], [], [], []

    events = sdf.get("events")
    relations = sdf.get("relations")
    if not isinstance(events, list):
        events = []
    if not isinstance(relations, list):
        relations = []

    limit_events = max(0, int(max_events))
    limit_rels = max(0, int(max_relations))
    if limit_events <= 0:
        return [], [], [], []

    doc_namespace = str(sdf.get("doc_namespace") or "").strip()
    if not doc_namespace:
        doc_namespace = "unknown_doc"

    provenance_chunk_ids = [chunk_id] if int(max_source_chunks) > 0 else []

    node_by_id: dict[str, dict[str, Any]] = {}
    has_subevent_edges: list[dict[str, Any]] = []
    before_edges: list[dict[str, Any]] = []
    chunk_links: list[dict[str, Any]] = []

    # 1) Event nodes + children edges (HAS_SUBEVENT)
    for event in events[:limit_events]:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("@id") or "").strip()
        name = str(event.get("name") or "").strip()
        if not event_id or not name:
            continue
        name_norm = text_processing(name)
        if not name_norm:
            continue

        temporal = event.get("temporal") if isinstance(event.get("temporal"), dict) else {}
        effective_date = str((temporal or {}).get("effective_date") or "").strip() or None
        valid_from = str((temporal or {}).get("valid_from") or "").strip() or None
        valid_to = str((temporal or {}).get("valid_to") or "").strip() or None

        attributes = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
        attributes_json = json.dumps(attributes, ensure_ascii=False, sort_keys=True)

        node_by_id.setdefault(
            event_id,
            {
                "sdf_event_id": event_id,
                "doc_namespace": doc_namespace,
                "name": name,
                "name_normalized": name_norm,
                "description": str(event.get("description") or "").strip() or None,
                "children_gate": str(event.get("children_gate") or "").strip() or None,
                "effective_date": effective_date,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "scope": str(event.get("scope") or "").strip() or None,
                "priority": event.get("priority"),
                "attributes_json": attributes_json,
                "source_chunk_ids": provenance_chunk_ids,
                "source_chunk_ids_truncated": False,
                "occurrences": 1,
                "owner_id": db_owner_id,
            },
        )

        chunk_links.append(
            {
                "chunk_id": chunk_id,
                "sdf_event_id": event_id,
                "owner_id": db_owner_id,
            }
        )

        children = event.get("children")
        if not isinstance(children, list):
            continue
        for child in children:
            if not isinstance(child, dict):
                continue
            child_id = str(child.get("child") or "").strip()
            if not child_id:
                continue
            imp = child.get("importance")
            importance = float(imp) if isinstance(imp, (int, float)) else None
            has_subevent_edges.append(
                {
                    "parent_id": event_id,
                    "child_id": child_id,
                    "importance": importance,
                    "doc_namespace": doc_namespace,
                    "source_chunk_ids": provenance_chunk_ids,
                    "source_chunk_ids_truncated": False,
                    "occurrences": 1,
                    "owner_id": db_owner_id,
                }
            )

    # 2) BEFORE relations
    for rel in relations[:limit_rels]:
        if not isinstance(rel, dict):
            continue
        if str(rel.get("wd_label") or "").strip().lower() not in {"before", ""}:
            continue
        left = str(rel.get("relationSubject") or "").strip()
        right = str(rel.get("relationObject") or "").strip()
        if not left or not right:
            continue
        before_edges.append(
            {
                "subject_id": left,
                "object_id": right,
                "doc_namespace": doc_namespace,
                "source_chunk_ids": provenance_chunk_ids,
                "source_chunk_ids_truncated": False,
                "occurrences": 1,
                "owner_id": db_owner_id,
            }
        )

    return list(node_by_id.values()), has_subevent_edges, before_edges, chunk_links

