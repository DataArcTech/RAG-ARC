from typing import Any, Dict, Optional

from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id


def _max_iso_datetime(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """
    Merge ISO-8601 datetime strings by taking the later one.

    Assumes both values are in a comparable ISO format (e.g. `...+00:00`).
    """
    if not a:
        return b
    if not b:
        return a
    return max(str(a), str(b))


def upsert_fact_occurrence(
    fact_data_by_id: Dict[str, Dict[str, Any]],
    *,
    head_id: str,
    head_name: str,
    relation_type: str,
    tail_id: str,
    tail_name: str,
    chunk_id: str,
    owner_id: Optional[str],
    db_owner_id: str,
    schema_version: str,
    domain: str,
    direction_sensitive: bool = True,
    max_source_chunks: int = 50,
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
    effective_date: Optional[str] = None,
) -> str:
    """
    Merge a (head, predicate, tail) occurrence into an aggregated payload used by Neo4j UNWIND insertion.

    - `fact_id` is owner-scoped and stable for the normalized triple text.
    - `occurrences` counts how many times the fact was extracted across input chunks.
    - `source_chunk_ids` keeps a de-duplicated list of chunk ids that produced the fact.
    """
    # For predicates that are NOT direction-sensitive, treat facts as undirected to avoid inverse duplicates
    # (A -[P]-> B and B -[P]-> A) amplifying noise in PPR/expansion/exports.
    if not bool(direction_sensitive) and tail_id < head_id:
        head_id, tail_id = tail_id, head_id
        head_name, tail_name = tail_name, head_name

    head_id = str(head_id or "").strip()
    tail_id = str(tail_id or "").strip()
    relation_type = str(relation_type or "").strip()
    fact_key = f"{head_id}|{relation_type}|{tail_id}"
    fact_id = compute_mdhash_id(fact_key, prefix="fact-", owner_id=owner_id)
    fact_text = str((head_name, relation_type, tail_name))

    existing = fact_data_by_id.get(fact_id)
    if existing is None:
        max_source_chunks = max(0, int(max_source_chunks))
        source_chunk_ids = [chunk_id] if max_source_chunks != 0 else []
        fact_data_by_id[fact_id] = {
            "fact_id": fact_id,
            "head_id": head_id,
            "tail_id": tail_id,
            "head_name": head_name,
            "relation_type": relation_type,
            "tail_name": tail_name,
            "fact_text": fact_text,
            "owner_id": db_owner_id,
            "source_chunk_ids": source_chunk_ids,
            # `occurrences` is the number of unique source chunks that asserted this fact (chunk-level evidence count).
            "occurrences": 1,
            # `mentions` is the raw extraction hit count (may exceed occurrences when the extractor emits duplicates).
            "mentions": 1,
            "schema_version": schema_version,
            "domain": domain,
            "source_chunk_ids_truncated": False,
            "direction_sensitive": bool(direction_sensitive),
            "valid_from": str(valid_from) if valid_from else None,
            "valid_to": str(valid_to) if valid_to else None,
            "effective_date": str(effective_date) if effective_date else (str(valid_from) if valid_from else None),
        }
        return fact_id

    existing["mentions"] = int(existing.get("mentions", existing.get("occurrences", 1))) + 1
    existing["valid_from"] = _max_iso_datetime(existing.get("valid_from"), valid_from)
    existing["valid_to"] = _max_iso_datetime(existing.get("valid_to"), valid_to)
    existing["effective_date"] = _max_iso_datetime(existing.get("effective_date"), effective_date or valid_from)
    max_source_chunks = max(0, int(max_source_chunks))
    source_chunks = list(existing.get("source_chunk_ids") or [])
    if chunk_id not in source_chunks:
        existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
        if max_source_chunks == 0:
            existing["source_chunk_ids_truncated"] = True
        elif len(source_chunks) < max_source_chunks:
            source_chunks.append(chunk_id)
            existing["source_chunk_ids"] = source_chunks
        else:
            existing["source_chunk_ids_truncated"] = True
    return fact_id
