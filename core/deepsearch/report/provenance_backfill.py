from typing import Any, Dict, List

from core.utils.json_extract import safe_json_loads

from config.core.deepsearch import report_composer_defaults as composer_defaults
from core.deepsearch.report.composer_helpers import _rank_evidences_for_question


def attach_provenance_chunk_evidence(
    *,
    graph_store: Any | None,
    trace: Dict[str, Any],
    evidences: List[Dict[str, Any]],
    question: str,
    max_attach: int,
) -> List[Dict[str, Any]]:
    """Backfill `source_chunk_ids` provenance into authoritative chunk evidences (Neo4j-only).

    Some deterministic tools (rule checks, joins, aggregations) reference `source_chunk_ids`
    in their provenance but may not include the underlying chunk contents in the evidence pool.
    This helper best-effort fetches those Chunk records from Neo4j and attaches them as extra
    evidences to improve citation grounding.
    """

    if not evidences or graph_store is None:
        return evidences
    execute = getattr(graph_store, "_execute_query", None)
    if not callable(execute):
        return evidences

    max_attach = int(max_attach or 0)
    if max_attach <= 0:
        return evidences

    graph_context = trace.get("graph_context") if isinstance(trace.get("graph_context"), dict) else {}
    access_scope = graph_context.get("access_scope") if isinstance(graph_context.get("access_scope"), dict) else {}
    scope_id = str(access_scope.get("scope_id") or "").strip()
    if not scope_id:
        return evidences

    owner_key = scope_id
    owner_key_fn = getattr(graph_store, "_owner_key", None)
    try:
        if callable(owner_key_fn):
            owner_key = str(owner_key_fn(scope_id))
    except Exception:
        owner_key = scope_id
    global_owner = getattr(graph_store, "OWNER_GLOBAL_KEY", "__GLOBAL__")

    existing_chunk_ids: set[str] = set()
    for item in evidences:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        content = str(item.get("content") or "").strip()
        if chunk_id and content and not any(
            str(chunk_id).startswith(prefix) for prefix in composer_defaults.TOOLISH_CHUNK_ID_PREFIXES
        ):
            existing_chunk_ids.add(chunk_id)

    requested: list[str] = []
    references: dict[str, list[dict[str, Any]]] = {}

    def _flatten(value: Any) -> list[str]:
        out: list[str] = []
        if value is None:
            return out
        if isinstance(value, str):
            token = value.strip()
            return [token] if token else out
        if isinstance(value, list):
            for item in value:
                out.extend(_flatten(item))
        return out

    def _remember(chunk_id: str, *, ref: dict[str, Any]) -> None:
        if not chunk_id or chunk_id in existing_chunk_ids:
            return
        if chunk_id not in references:
            references[chunk_id] = []
            requested.append(chunk_id)
        references[chunk_id].append(ref)

    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        prov = evidence.get("provenance") if isinstance(evidence.get("provenance"), dict) else {}
        if not prov:
            continue
        chunk_ids = []
        for key in ("source_chunk_ids", "left_source_chunk_ids", "right_source_chunk_ids"):
            chunk_ids.extend(_flatten(prov.get(key)))
        chunk_ids = [str(x).strip() for x in chunk_ids if str(x).strip()]
        if not chunk_ids:
            continue
        ref = {
            "evidence_id": str(evidence.get("chunk_id") or "").strip() or None,
            "source": str(evidence.get("source") or "").strip() or None,
            "fact_id": prov.get("fact_id"),
        }
        for cid in chunk_ids:
            _remember(cid, ref=ref)
            if len(requested) >= max_attach:
                break
        if len(requested) >= max_attach:
            break

    if not requested:
        return evidences

    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $chunk_ids
      AND COALESCE(c.owner_id, $global_owner) = $owner_id
    RETURN c.chunk_id AS chunk_id,
           c.content AS content,
           c.metadata AS metadata
    """
    try:
        rows = execute(query, {"chunk_ids": requested, "owner_id": owner_key, "global_owner": global_owner})
    except Exception:
        return evidences
    if not isinstance(rows, list) or not rows:
        return evidences

    attached: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        content = str(row.get("content") or "")
        if not chunk_id or not content or chunk_id in existing_chunk_ids:
            continue
        meta_raw = row.get("metadata")
        meta = safe_json_loads(meta_raw, expected="dict") if isinstance(meta_raw, str) else meta_raw
        if not isinstance(meta, dict):
            meta = {}
        attached.append(
            {
                "chunk_id": chunk_id,
                "source": "neo4j_chunk",
                "content": content,
                "score": 1.0,
                "provenance": {
                    "metadata": meta,
                    "linked_from": references.get(chunk_id, []),
                    "scope_id": scope_id,
                    "owner_id": owner_key,
                },
            }
        )
        if len(attached) >= max_attach:
            break

    if not attached:
        return evidences
    merged = list(evidences) + attached
    return _rank_evidences_for_question(question, merged)

