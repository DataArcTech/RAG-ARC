"""Cypher templates for identity/disambiguation/alignment maintenance.

Design:
- Identity nodes are owner-scoped and upserted by `identity_id`.
- We keep temporal/version fields primarily on mentions; identity holds only summary fields.
- Relationship history is preserved by "valid_to" semantics (Graphiti-style reconciliation):
  - old edges are not deleted; they are superseded (valid_to set) and new edges are created.
"""

UPSERT_ENTITY_IDENTITIES_QUERY = """
UNWIND $identities AS i
MERGE (n:EntityIdentity {identity_id: i.identity_id})
SET n.owner_id = i.owner_id,
    n.entity_type_key = i.entity_type_key,
    n.surface_entity_id = i.surface_entity_id,
    n.status = COALESCE(i.status, n.status, 'active'),
    n.centroid_dim = i.centroid_dim,
    n.centroid_b64 = i.centroid_b64,
    n.centroid_sum_b64 = i.centroid_sum_b64,
    n.member_count = i.member_count,
    n.valid_from_min = i.valid_from_min,
    n.valid_to_max = i.valid_to_max,
    n.source_versions = i.source_versions,
    n.updated_at = datetime(),
    n.created_at = COALESCE(n.created_at, datetime())
WITH n, i
MATCH (e:Entity {entity_id: i.surface_entity_id, owner_id: i.owner_id})
MERGE (n)-[r:HAS_SURFACE {surface_entity_id: i.surface_entity_id}]->(e)
SET r.owner_id = i.owner_id,
    r.updated_at = datetime(),
    r.created_at = COALESCE(r.created_at, datetime())
"""

# Supersede old RESOLVED_TO edges for mentions in scope (valid_to set).
SUPERSEDE_RESOLVED_TO_FOR_MENTIONS_QUERY = """
UNWIND $mention_ids AS mention_id
MATCH (m:EntityMention {mention_id: mention_id})-[r:RESOLVED_TO]->(:EntityIdentity)
WHERE r.valid_to IS NULL
SET r.valid_to = datetime(),
    r.superseded_by_run_id = $run_id,
    r.updated_at = datetime()
"""

SUPERSEDE_RESOLVED_TO_FOR_ASSIGNMENTS_QUERY = """
UNWIND $assignments AS a
MATCH (m:EntityMention {mention_id: a.mention_id, owner_id: a.owner_id})-[r:RESOLVED_TO {identity_id: a.identity_id}]->(:EntityIdentity)
WHERE r.valid_to IS NULL
SET r.valid_to = datetime(),
    r.superseded_by_run_id = $run_id,
    r.updated_at = datetime()
"""

UPSERT_RESOLVED_TO_QUERY = """
UNWIND $assignments AS a
MATCH (m:EntityMention {mention_id: a.mention_id, owner_id: a.owner_id})
MATCH (i:EntityIdentity {identity_id: a.identity_id})
MERGE (m)-[r:RESOLVED_TO {identity_id: a.identity_id}]->(i)
SET r.owner_id = a.owner_id,
    r.run_id = $run_id,
    r.method = a.method,
    r.confidence = a.confidence,
    r.valid_from = datetime(),
    r.valid_to = NULL,
    r.updated_at = datetime(),
    r.created_at = COALESCE(r.created_at, datetime())
"""

# Update centroid/time/member_count aggregates for identities (best-effort repair/recompute).
UPDATE_ENTITY_IDENTITIES_AGGREGATES_QUERY = """
UNWIND $updates AS u
MATCH (i:EntityIdentity {identity_id: u.identity_id, owner_id: u.owner_id})
SET i.centroid_dim = u.centroid_dim,
    i.centroid_b64 = u.centroid_b64,
    i.centroid_sum_b64 = u.centroid_sum_b64,
    i.member_count = u.member_count,
    i.valid_from_min = u.valid_from_min,
    i.valid_to_max = u.valid_to_max,
    i.updated_at = datetime()
"""

# Mark active identities as inactive when they have no active RESOLVED_TO evidence.
MARK_ORPHAN_IDENTITIES_INACTIVE_QUERY = """
MATCH (i:EntityIdentity {owner_id: $owner_id})
WHERE coalesce(i.status, 'active') = 'active'
OPTIONAL MATCH (:EntityMention {owner_id: $owner_id})-[r:RESOLVED_TO]->(i)
WHERE r.valid_to IS NULL
WITH i, count(r) AS c
WHERE c = 0
SET i.status = 'inactive',
    i.updated_at = datetime()
RETURN count(i) AS marked
"""

# Repair aggregates for orphan identities (no active RESOLVED_TO edges).
# We keep centroid/time summary as historical, but `member_count` should reflect active evidence (0).
ZERO_ORPHAN_IDENTITIES_MEMBER_COUNT_QUERY = """
MATCH (i:EntityIdentity {owner_id: $owner_id})
WHERE coalesce(i.status, 'active') <> 'active'
OPTIONAL MATCH (:EntityMention {owner_id: $owner_id})-[r:RESOLVED_TO]->(i)
WHERE r.valid_to IS NULL
WITH i, count(r) AS c
WHERE c = 0
SET i.member_count = 0,
    i.updated_at = datetime()
RETURN count(i) AS zeroed
"""

# Supersede SAME_AS edges pointing to inactive identities.
SUPERSEDE_SAME_AS_FOR_INACTIVE_IDENTITIES_QUERY = """
MATCH (a:EntityIdentity {owner_id: $owner_id})-[r:SAME_AS]->(b:EntityIdentity {owner_id: $owner_id})
WHERE r.valid_to IS NULL
  AND (coalesce(a.status, 'active') <> 'active' OR coalesce(b.status, 'active') <> 'active')
SET r.valid_to = datetime(),
    r.superseded_by_run_id = $run_id,
    r.updated_at = datetime()
RETURN count(r) AS superseded
"""

# Supersede all active SAME_AS edges for this owner (used by L2 rebuild).
SUPERSEDE_ALL_SAME_AS_FOR_OWNER_QUERY = """
MATCH (:EntityIdentity {owner_id: $owner_id})-[r:SAME_AS]->(:EntityIdentity {owner_id: $owner_id})
WHERE r.valid_to IS NULL
SET r.valid_to = datetime(),
    r.superseded_by_run_id = $run_id,
    r.updated_at = datetime()
RETURN count(r) AS superseded
"""

# SAME_AS is stored in canonical direction to avoid duplicates: left_id < right_id.
UPSERT_SAME_AS_QUERY = """
UNWIND $pairs AS p
MATCH (a:EntityIdentity {identity_id: p.left_id})
MATCH (b:EntityIdentity {identity_id: p.right_id})
MERGE (a)-[r:SAME_AS {right_id: p.right_id}]->(b)
SET r.owner_id = p.owner_id,
    r.run_id = $run_id,
    r.score = p.score,
    r.method = p.method,
    r.time_overlap = p.time_overlap,
    r.reason = p.reason,
    r.valid_from = datetime(),
    r.valid_to = NULL,
    r.updated_at = datetime(),
    r.created_at = COALESCE(r.created_at, datetime())
"""
