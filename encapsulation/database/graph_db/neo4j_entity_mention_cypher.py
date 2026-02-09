"""Cypher templates for entity-mention (occurrence) materialization.

This is used by:
- L0 hot-path mention materialization during ingestion (no extra read queries).
- Background "fill missing mentions" jobs (query MENTIONS then upsert mentions).

Design notes
------------
- Mention nodes are owner-scoped and idempotent via `mention_id`.
- We keep temporal/version fields as *strings* to avoid transaction aborts on parse errors.
  (Downstream can parse to Neo4j datetime when needed.)
"""

UPSERT_ENTITY_MENTIONS_QUERY = """
UNWIND $entity_mentions AS m
MATCH (c:Chunk {chunk_id: m.chunk_id, owner_id: m.owner_id})
MATCH (e:Entity {entity_id: m.surface_entity_id, owner_id: m.owner_id})
MERGE (em:EntityMention {mention_id: m.mention_id})
SET em.owner_id = m.owner_id,
    em.chunk_id = m.chunk_id,
    em.surface_entity_id = m.surface_entity_id,
    em.source_file_id = m.source_file_id,
    em.source_version = m.source_version,
    em.valid_from = m.valid_from,
    em.valid_to = m.valid_to,
    em.effective_date = m.effective_date,
    em.updated_at = datetime(),
    em.created_at = COALESCE(em.created_at, datetime())
MERGE (c)-[r1:HAS_MENTION {mention_id: m.mention_id}]->(em)
SET r1.owner_id = m.owner_id,
    r1.updated_at = datetime(),
    r1.created_at = COALESCE(r1.created_at, datetime())
MERGE (em)-[r2:MENTIONS_SURFACE {surface_entity_id: m.surface_entity_id}]->(e)
SET r2.owner_id = m.owner_id,
    r2.updated_at = datetime(),
    r2.created_at = COALESCE(r2.created_at, datetime())
"""
