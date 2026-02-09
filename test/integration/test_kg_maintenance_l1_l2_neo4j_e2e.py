import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pytest

from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig


def _env_ready_neo4j() -> bool:
    # Defaults exist for URL/username/database, but password is typically required.
    return bool(str(os.getenv("NEO4J_PASSWORD", "")).strip())


@pytest.fixture(scope="session")
def graph_store():  # noqa: ANN001
    if os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test opt-in: set RUN_RAGARC_INTEGRATION_TESTS=1")
    if not _env_ready_neo4j():
        pytest.skip("integration test requires Neo4j env (NEO4J_PASSWORD at least)")

    # Use local embedding config (no external API) so KG maintenance tests are deterministic.
    cfg = PrunedHippoRAGNeo4jConfig(
        url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        embedding=QwenEmbeddingConfig(),
        shared_instance=False,
        # KG maintenance is deterministic; keep synonymy edges as-is (not used here).
    )
    try:
        return cfg.build()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j store init failed: {exc}")


def _cypher(store, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:  # noqa: ANN001
    rows = store._execute_query(str(query), dict(params or {}))  # noqa: SLF001
    return list(rows or [])


def _cleanup_owner(store, owner_id: str) -> None:  # noqa: ANN001
    _cypher(store, "MATCH (n) WHERE n.owner_id = $owner_id DETACH DELETE n", {"owner_id": owner_id})


def _count(store, query: str, params: Dict[str, Any]) -> int:  # noqa: ANN001
    rows = _cypher(store, query, params)
    if not rows:
        return 0
    try:
        return int(rows[0].get("n") or 0)
    except Exception:
        return 0


def _make_chunks_and_surface(
    store,  # noqa: ANN001
    *,
    owner_id: str,
    file_id: str,
    chunk_specs: List[Tuple[str, Dict[str, Any]]],
    surface_entity_id: str,
    entity_type_key: str = "entity",
) -> List[str]:
    # Create a surface entity + chunk nodes + MENTIONS edges.
    _cypher(
        store,
        """
        MERGE (e:Entity {owner_id:$owner_id, entity_id:$entity_id})
        SET e.entity_type_key = $entity_type_key
        """,
        {"owner_id": owner_id, "entity_id": surface_entity_id, "entity_type_key": entity_type_key},
    )

    chunk_ids: List[str] = []
    for chunk_id, meta in chunk_specs:
        payload = {
            "owner_id": owner_id,
            "chunk_id": chunk_id,
            "source_file_id": file_id,
            "metadata": json.dumps(meta, ensure_ascii=False),
        }
        _cypher(
            store,
            """
            MERGE (c:Chunk {owner_id:$owner_id, chunk_id:$chunk_id})
            SET c.source_file_id = $source_file_id,
                c.metadata = $metadata
            WITH c
            MATCH (e:Entity {owner_id:$owner_id, entity_id:$entity_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            {**payload, "entity_id": surface_entity_id},
        )
        chunk_ids.append(chunk_id)
    return chunk_ids


def test_kg_maintenance_l1_disambiguates_same_surface_by_context(graph_store) -> None:  # noqa: ANN001
    """
    E2E-ish: write minimal Chunk-[:MENTIONS]->Entity edges into Neo4j, then run:
    - L0 backfill (materialize EntityMention)
    - L1 disambiguation (EntityIdentity + RESOLVED_TO)

    Boundary: same surface string should split into multiple identities when chunk embeddings form
    well-separated clusters (without any LLM calls).
    """
    owner_id = str(uuid.uuid4())
    file_id = f"file-{uuid.uuid4().hex}"
    try:
        # Two tight clusters in embedding space.
        v_a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        v_b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        chunk_specs = [
            ("c-a1", {"business_time": {"valid_from": "2024-01-01"}}),
            ("c-a2", {"business_time": {"valid_from": "2024-01-02"}}),
            ("c-b1", {"business_time": {"valid_from": "2024-06-01"}}),
            ("c-b2", {"business_time": {"valid_from": "2024-06-02"}}),
        ]
        chunk_ids = _make_chunks_and_surface(
            graph_store,
            owner_id=owner_id,
            file_id=file_id,
            chunk_specs=chunk_specs,
            surface_entity_id="entity-surface-1",
            entity_type_key="organization",
        )

        # Inject chunk embeddings (maintenance reads from in-memory dict).
        graph_store.chunk_embeddings["c-a1"] = v_a
        graph_store.chunk_embeddings["c-a2"] = v_a
        graph_store.chunk_embeddings["c-b1"] = v_b
        graph_store.chunk_embeddings["c-b2"] = v_b

        l0 = graph_store.materialize_entity_mentions_for_chunk_ids(owner_id=owner_id, chunk_ids=chunk_ids)
        assert l0.get("success") is True
        assert int(l0.get("written") or 0) == 4

        l1 = graph_store.run_kg_maintenance_l1_for_file_ids(owner_id=owner_id, file_ids=[file_id])
        assert l1.get("success") is True
        assert int((l1.get("counts") or {}).get("mentions_total") or 0) == 4

        identities = _count(
            graph_store,
            "MATCH (i:EntityIdentity {owner_id:$owner_id}) RETURN count(i) AS n",
            {"owner_id": owner_id},
        )
        # With two orthogonal clusters and min_sim default~=0.8, we expect 2 identities.
        assert identities == 2

        resolved = _count(
            graph_store,
            """
            MATCH (:EntityMention {owner_id:$owner_id})-[r:RESOLVED_TO]->(:EntityIdentity {owner_id:$owner_id})
            WHERE r.valid_to IS NULL
            RETURN count(r) AS n
            """,
            {"owner_id": owner_id},
        )
        assert resolved == 4
    finally:
        for cid in ("c-a1", "c-a2", "c-b1", "c-b2"):
            try:
                graph_store.chunk_embeddings.pop(cid, None)
            except Exception:
                pass
        _cleanup_owner(graph_store, owner_id)


def test_kg_maintenance_l2_repairs_duplicate_assignments_and_orphans(graph_store) -> None:  # noqa: ANN001
    """
    E2E-ish: craft broken graph state, then run L2:
    - A mention has 2 active RESOLVED_TO edges (duplicate assignment)
    - Orphan identities exist (no active assignments), but member_count is stale
    L2 should:
    - supersede the lower-confidence duplicate edge
    - mark orphans inactive and zero their member_count
    - bring member_count drift to 0
    """
    owner_id = str(uuid.uuid4())
    file_id = f"file-{uuid.uuid4().hex}"
    try:
        # Minimal surface + chunk + mention.
        chunk_ids = _make_chunks_and_surface(
            graph_store,
            owner_id=owner_id,
            file_id=file_id,
            chunk_specs=[("c-x1", {"business_time": {"valid_from": "2024-01-01"}})],
            surface_entity_id="entity-surface-x",
            entity_type_key="policy",
        )
        graph_store.chunk_embeddings["c-x1"] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        l0 = graph_store.materialize_entity_mentions_for_chunk_ids(owner_id=owner_id, chunk_ids=chunk_ids)
        assert l0.get("success") is True

        # Create two identities and attach two active RESOLVED_TO edges to the same mention.
        mention_rows = _cypher(
            graph_store,
            "MATCH (m:EntityMention {owner_id:$owner_id}) RETURN m.mention_id AS mention_id LIMIT 1",
            {"owner_id": owner_id},
        )
        assert mention_rows and mention_rows[0].get("mention_id")
        mention_id = str(mention_rows[0]["mention_id"])

        _cypher(
            graph_store,
            """
            CREATE (i1:EntityIdentity {owner_id:$owner_id, identity_id:$i1, surface_entity_id:$surface, entity_type_key:'policy', status:'active', member_count: 99})
            CREATE (i2:EntityIdentity {owner_id:$owner_id, identity_id:$i2, surface_entity_id:$surface, entity_type_key:'policy', status:'active', member_count: 77})
            WITH i1, i2
            MATCH (m:EntityMention {owner_id:$owner_id, mention_id:$mention_id})
            CREATE (m)-[:RESOLVED_TO {owner_id:$owner_id, identity_id:$i1, confidence: 0.9, valid_from: datetime(), valid_to: NULL}]->(i1)
            CREATE (m)-[:RESOLVED_TO {owner_id:$owner_id, identity_id:$i2, confidence: 0.1, valid_from: datetime(), valid_to: NULL}]->(i2)
            """,
            {
                "owner_id": owner_id,
                "mention_id": mention_id,
                "surface": "entity-surface-x",
                "i1": "identity-high",
                "i2": "identity-low",
            },
        )

        # Add a fully orphan identity with stale member_count.
        _cypher(
            graph_store,
            "CREATE (:EntityIdentity {owner_id:$owner_id, identity_id:$iid, surface_entity_id:'entity-orphan', entity_type_key:'policy', status:'active', member_count: 123})",
            {"owner_id": owner_id, "iid": "identity-orphan"},
        )

        drift_before = _count(
            graph_store,
            """
            MATCH (i:EntityIdentity {owner_id:$owner_id})
            OPTIONAL MATCH (:EntityMention {owner_id:$owner_id})-[r:RESOLVED_TO]->(i)
            WHERE r.valid_to IS NULL
            WITH i, count(r) AS c
            WHERE coalesce(i.member_count, 0) <> c
            RETURN count(i) AS n
            """,
            {"owner_id": owner_id},
        )
        assert drift_before >= 1

        l2 = graph_store.run_kg_maintenance_l2_for_owner(owner_id=owner_id, rebuild_same_as=False)
        assert l2.get("success") is True

        # Only one active RESOLVED_TO edge should remain.
        active_edges = _count(
            graph_store,
            """
            MATCH (:EntityMention {owner_id:$owner_id, mention_id:$mention_id})-[r:RESOLVED_TO]->(:EntityIdentity {owner_id:$owner_id})
            WHERE r.valid_to IS NULL
            RETURN count(r) AS n
            """,
            {"owner_id": owner_id, "mention_id": mention_id},
        )
        assert active_edges == 1

        # Orphan member_count should be repaired to 0 and drift should be 0.
        orphan_mc_rows = _cypher(
            graph_store,
            "MATCH (i:EntityIdentity {owner_id:$owner_id, identity_id:'identity-orphan'}) RETURN coalesce(i.member_count, -1) AS mc, coalesce(i.status,'') AS st",
            {"owner_id": owner_id},
        )
        assert orphan_mc_rows
        assert int(orphan_mc_rows[0].get("mc")) == 0
        assert str(orphan_mc_rows[0].get("st")) == "inactive"

        drift_after = _count(
            graph_store,
            """
            MATCH (i:EntityIdentity {owner_id:$owner_id})
            OPTIONAL MATCH (:EntityMention {owner_id:$owner_id})-[r:RESOLVED_TO]->(i)
            WHERE r.valid_to IS NULL
            WITH i, count(r) AS c
            WHERE coalesce(i.member_count, 0) <> c
            RETURN count(i) AS n
            """,
            {"owner_id": owner_id},
        )
        assert drift_after == 0
    finally:
        graph_store.chunk_embeddings.pop("c-x1", None)
        _cleanup_owner(graph_store, owner_id)


def test_kg_maintenance_l2_is_owner_scoped(graph_store) -> None:  # noqa: ANN001
    """
    Boundary: multi-tenant isolation.
    Running L2 for one owner must not modify identities of another owner.
    """
    owner_a = str(uuid.uuid4())
    owner_b = str(uuid.uuid4())
    try:
        _cypher(
            graph_store,
            "CREATE (:EntityIdentity {owner_id:$owner_id, identity_id:'i-a', surface_entity_id:'s-a', entity_type_key:'entity', status:'active', member_count: 11})",
            {"owner_id": owner_a},
        )
        _cypher(
            graph_store,
            "CREATE (:EntityIdentity {owner_id:$owner_id, identity_id:'i-b', surface_entity_id:'s-b', entity_type_key:'entity', status:'active', member_count: 22})",
            {"owner_id": owner_b},
        )

        graph_store.run_kg_maintenance_l2_for_owner(owner_id=owner_a, rebuild_same_as=False)

        rows_a = _cypher(
            graph_store,
            "MATCH (i:EntityIdentity {owner_id:$owner_id, identity_id:'i-a'}) RETURN coalesce(i.member_count,-1) AS mc, coalesce(i.status,'') AS st",
            {"owner_id": owner_a},
        )
        assert rows_a and int(rows_a[0].get("mc")) == 0
        assert str(rows_a[0].get("st")) == "inactive"

        rows_b = _cypher(
            graph_store,
            "MATCH (i:EntityIdentity {owner_id:$owner_id, identity_id:'i-b'}) RETURN coalesce(i.member_count,-1) AS mc, coalesce(i.status,'') AS st",
            {"owner_id": owner_b},
        )
        assert rows_b and int(rows_b[0].get("mc")) == 22
        assert str(rows_b[0].get("st")) == "active"
    finally:
        _cleanup_owner(graph_store, owner_a)
        _cleanup_owner(graph_store, owner_b)


def test_kg_maintenance_l1_alignment_emits_time_overlap_signal(graph_store) -> None:  # noqa: ANN001
    """
    Boundary: time/version.
    When two identities are embedding-similar but have decisively non-overlapping time windows,
    alignment should still be allowed to emit SAME_AS, but must surface `time_overlap=false`
    and a clear `reason` (so downstream can avoid irreversible merges).
    """
    owner_id = str(uuid.uuid4())
    file_id = f"file-{uuid.uuid4().hex}"
    try:
        # Two different surfaces with identical embeddings, but disjoint time windows.
        chunk_specs_1 = [("c-t1", {"business_time": {"valid_from": "2024-01-01", "valid_to": "2024-01-31"}})]
        chunk_specs_2 = [("c-t2", {"business_time": {"valid_from": "2025-01-01", "valid_to": "2025-01-31"}})]
        _make_chunks_and_surface(
            graph_store,
            owner_id=owner_id,
            file_id=file_id,
            chunk_specs=chunk_specs_1,
            surface_entity_id="entity-surface-t1",
            entity_type_key="policy",
        )
        _make_chunks_and_surface(
            graph_store,
            owner_id=owner_id,
            file_id=file_id,
            chunk_specs=chunk_specs_2,
            surface_entity_id="entity-surface-t2",
            entity_type_key="policy",
        )

        v = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        graph_store.chunk_embeddings["c-t1"] = v
        graph_store.chunk_embeddings["c-t2"] = v

        l0 = graph_store.materialize_entity_mentions_for_chunk_ids(owner_id=owner_id, chunk_ids=["c-t1", "c-t2"])
        assert l0.get("success") is True
        assert int(l0.get("written") or 0) == 2

        l1 = graph_store.run_kg_maintenance_l1_for_file_ids(owner_id=owner_id, file_ids=[file_id])
        assert l1.get("success") is True

        # Alignment should emit exactly one SAME_AS edge between the two identities.
        rows = _cypher(
            graph_store,
            """
            MATCH (:EntityIdentity {owner_id:$owner_id})-[r:SAME_AS]->(:EntityIdentity {owner_id:$owner_id})
            WHERE r.valid_to IS NULL
            RETURN coalesce(r.time_overlap, 'null') AS time_overlap, coalesce(r.reason, '') AS reason
            """,
            {"owner_id": owner_id},
        )
        assert rows, "expected at least one SAME_AS edge"
        # There can be >1 edges if other tests ran without cleanup; ensure at least one carries the no-overlap signal.
        assert any((str(r.get("time_overlap")).lower() == "false") for r in rows)
        assert any("no_time_overlap" in str(r.get("reason") or "") for r in rows)
    finally:
        graph_store.chunk_embeddings.pop("c-t1", None)
        graph_store.chunk_embeddings.pop("c-t2", None)
        _cleanup_owner(graph_store, owner_id)
