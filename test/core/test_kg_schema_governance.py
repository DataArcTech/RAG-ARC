from core.graph_adapter.hipporag import HippoRAGGraphAdapter
from core.knowledge_graph.schema import schema_from_dict
from encapsulation.database.utils.fact_provenance import upsert_fact_occurrence


def test_schema_normalization_alias_and_allowlist_reject_unknown() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {
                "default": {
                    "allowed_relations": ["LOCATED_IN"],
                    "relation_aliases": {"located in": "LOCATED_IN"},
                    "unknown_predicate_policy": "reject",
                }
            },
        }
    )
    domain_schema = schema.for_domain("default")

    assert domain_schema.normalize_predicate("located in") == "LOCATED_IN"
    assert domain_schema.normalize_predicate("LOCATED IN") == "LOCATED_IN"
    assert domain_schema.normalize_predicate("random_relation") is None


def test_schema_unknown_policy_collapse_to_related() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {
                "default": {
                    "allowed_relations": ["RELATED_TO"],
                    "relation_aliases": {},
                    "unknown_predicate_policy": "collapse",
                    "unknown_predicate_fallback": "RELATED_TO",
                }
            },
        }
    )
    domain_schema = schema.for_domain("default")
    assert domain_schema.normalize_predicate("some free-form predicate") == "RELATED_TO"


def test_schema_direction_sensitive_relations_union() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {
                "default": {"direction_sensitive_relations": ["OWNS"]},
                "finance": {"direction_sensitive_relations": ["PART_OF"]},
            },
        }
    )
    assert schema.direction_sensitive_relations_all() == {"OWNS", "PART_OF"}


def test_fact_provenance_upsert_merges_occurrences_and_sources() -> None:
    facts: dict[str, dict] = {}
    fact_id_1 = upsert_fact_occurrence(
        facts,
        head_id="entity-apple",
        head_name="apple inc",
        relation_type="OWNS",
        tail_id="entity-beats",
        tail_name="beats",
        chunk_id="chunk-1",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
    )
    fact_id_2 = upsert_fact_occurrence(
        facts,
        head_id="entity-apple",
        head_name="apple inc",
        relation_type="OWNS",
        tail_id="entity-beats",
        tail_name="beats",
        chunk_id="chunk-2",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
    )

    assert fact_id_1 == fact_id_2
    payload = facts[fact_id_1]
    assert payload["occurrences"] == 2
    assert set(payload["source_chunk_ids"]) == {"chunk-1", "chunk-2"}

    capped: dict[str, dict] = {}
    upsert_fact_occurrence(
        capped,
        head_id="entity-apple",
        head_name="apple inc",
        relation_type="OWNS",
        tail_id="entity-beats",
        tail_name="beats",
        chunk_id="chunk-1",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        max_source_chunks=1,
    )
    upsert_fact_occurrence(
        capped,
        head_id="entity-apple",
        head_name="apple inc",
        relation_type="OWNS",
        tail_id="entity-beats",
        tail_name="beats",
        chunk_id="chunk-2",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        max_source_chunks=1,
    )
    payload2 = next(iter(capped.values()))
    assert payload2["occurrences"] == 2
    assert payload2["source_chunk_ids"] == ["chunk-1"]
    assert payload2["source_chunk_ids_truncated"] is True


def test_fact_provenance_canonicalizes_inverse_facts_for_non_direction_sensitive_predicates() -> None:
    facts: dict[str, dict] = {}
    fact_id_1 = upsert_fact_occurrence(
        facts,
        head_id="entity-alpha",
        head_name="alpha",
        relation_type="HAS_FEATURE",
        tail_id="entity-beta",
        tail_name="beta",
        chunk_id="chunk-1",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        direction_sensitive=False,
    )
    fact_id_2 = upsert_fact_occurrence(
        facts,
        head_id="entity-beta",
        head_name="beta",
        relation_type="HAS_FEATURE",
        tail_id="entity-alpha",
        tail_name="alpha",
        chunk_id="chunk-2",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        direction_sensitive=False,
    )
    assert fact_id_1 == fact_id_2
    payload = facts[fact_id_1]
    assert payload["occurrences"] == 2
    assert payload["head_id"] <= payload["tail_id"]
    assert set(payload["source_chunk_ids"]) == {"chunk-1", "chunk-2"}


def test_fact_provenance_keeps_direction_for_direction_sensitive_predicates() -> None:
    facts: dict[str, dict] = {}
    fact_id_1 = upsert_fact_occurrence(
        facts,
        head_id="entity-alpha",
        head_name="alpha",
        relation_type="OWNS",
        tail_id="entity-beta",
        tail_name="beta",
        chunk_id="chunk-1",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        direction_sensitive=True,
    )
    fact_id_2 = upsert_fact_occurrence(
        facts,
        head_id="entity-beta",
        head_name="beta",
        relation_type="OWNS",
        tail_id="entity-alpha",
        tail_name="alpha",
        chunk_id="chunk-2",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        direction_sensitive=True,
    )
    assert fact_id_1 != fact_id_2


def test_beam_search_paths_respect_direction() -> None:
    relations = [
        {"head": "A", "relation": "OWNS", "tail": "B", "directed": True, "weight": 1.0},
        {"head": "B", "relation": "OWNS", "tail": "C", "directed": True, "weight": 1.0},
    ]
    paths = HippoRAGGraphAdapter._beam_search_paths(relations, seeds=["A", "C"], beam_size=3, max_depth=3)
    assert any(path.get("nodes") == ["A", "B", "C"] for path in paths)
