"""
Neo4j chunk upsert cleanup helpers for PrunedHippoRAG.

Why this exists
---------------
The Neo4j ingest path aggregates evidence on `(:Entity)-[:RELATES_TO].source_chunk_ids` and increments
`occurrences/weight` in an append-only fashion. When a chunk is *re-indexed with the same chunk_id*,
we must first remove the old evidence contributed by that chunk_id; otherwise the graph drifts over time
and PPR is biased by stale facts.

This helper runs *best-effort* cleanup queries inside the same write transaction prior to inserting the
new graph evidence.
"""
from typing import Any, Mapping, Sequence


def run_chunk_replace_cleanup(
    tx: Any,
    *,
    chunk_keys: Sequence[Mapping[str, str]],
) -> None:
    """
    Cleanup old graph evidence for chunk upserts.

    Parameters
    ----------
    tx:
        An open Neo4j transaction (must expose `run(query, params)`).
    chunk_keys:
        Sequence of mappings: {"chunk_id": <str>, "owner_id": <db_owner_id str>}.

    Notes
    -----
    - Chunk nodes are MERGEd by chunk_id in the current schema; owner_id is used as a guard.
    - Facts (RELATES_TO) are owner-scoped via r.owner_id.
    """
    keys = []
    for item in chunk_keys or []:
        chunk_id = str((item or {}).get("chunk_id") or "").strip()
        owner_id = str((item or {}).get("owner_id") or "").strip()
        if chunk_id and owner_id:
            keys.append({"chunk_id": chunk_id, "owner_id": owner_id})
    if not keys:
        return

    # 1) Remove chunk -> entity mention edges for these chunks so they can be rebuilt.
    tx.run(
        """
        UNWIND $chunk_keys AS k
        MATCH (c:Chunk {chunk_id: k.chunk_id})
        WHERE COALESCE(c.owner_id, k.owner_id) = k.owner_id
        MATCH (c)-[r:MENTIONS]->(:Entity)
        WHERE COALESCE(r.owner_id, k.owner_id) = k.owner_id
        DELETE r
        """,
        {"chunk_keys": keys},
    )

    # 1.5) Remove Section -> Chunk links so they can be rebuilt.
    tx.run(
        """
        UNWIND $chunk_keys AS k
        MATCH (c:Chunk {chunk_id: k.chunk_id})
        WHERE COALESCE(c.owner_id, k.owner_id) = k.owner_id
        MATCH (:Section)-[r:HAS_CHUNK]->(c)
        WHERE COALESCE(r.owner_id, k.owner_id) = k.owner_id
        DELETE r
        """,
        {"chunk_keys": keys},
    )

    # 1.6) Remove TreeNode -> Chunk links so they can be rebuilt.
    tx.run(
        """
        UNWIND $chunk_keys AS k
        MATCH (c:Chunk {chunk_id: k.chunk_id})
        WHERE COALESCE(c.owner_id, k.owner_id) = k.owner_id
        MATCH (t:TreeNode)-[r:HAS_CHUNK]->(c)
        WHERE COALESCE(t.owner_id, k.owner_id) = k.owner_id
        DELETE r
        """,
        {"chunk_keys": keys},
    )

    # 1.7) Cleanup orphaned TreeNodes (no remaining HAS_CHUNK).
    tx.run(
        """
        UNWIND $chunk_keys AS k
        MATCH (t:TreeNode {owner_id: k.owner_id})
        WHERE NOT (t)-[:HAS_CHUNK]->(:Chunk)
        DETACH DELETE t
        """,
        {"chunk_keys": keys},
    )

    # 2) Remove stale provenance from aggregated fact edges.
    #    If a fact loses all evidence after filtering, delete the relationship.
    #
    #    occurrences/weight:
    #    - When `source_chunk_ids_truncated` is false, occurrences can be recomputed from source_chunk_ids length.
    #    - When truncated, we cannot know the true evidence count; keep occurrences as-is (best effort).
    tx.run(
        """
        UNWIND $chunk_keys AS k
        MATCH ()-[r:RELATES_TO]-()
        WHERE COALESCE(r.owner_id, k.owner_id) = k.owner_id
          AND k.chunk_id IN COALESCE(r.source_chunk_ids, [])
        WITH r, k, [cid IN COALESCE(r.source_chunk_ids, []) WHERE cid <> k.chunk_id] AS filtered
        SET r.source_chunk_ids = filtered,
            r.occurrences = CASE
                WHEN COALESCE(r.source_chunk_ids_truncated, false) THEN COALESCE(r.occurrences, size(filtered))
                ELSE size(filtered)
            END,
            r.weight = toFloat(
                CASE
                    WHEN COALESCE(r.source_chunk_ids_truncated, false) THEN COALESCE(r.occurrences, size(filtered))
                    ELSE size(filtered)
                END
            ),
            r.updated_at = datetime()
        WITH r
        WHERE size(COALESCE(r.source_chunk_ids, [])) = 0
        DELETE r
        """,
        {"chunk_keys": keys},
    )
