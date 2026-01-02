from encapsulation.database.graph_db.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jStore


def test_neo4j_schema_statements_cover_cypher_tool_access_patterns() -> None:
    ddl = PrunedHippoRAGNeo4jStore.__wrapped__.neo4j_schema_statements()
    joined = "\n".join(ddl)

    # Cypher tools locate entities by (owner_id, entity_name_normalized) and filter by entity_type.
    assert "entity_owner_name_normalized IF NOT EXISTS" in joined
    assert "entity_owner_type IF NOT EXISTS" in joined

    # Cypher tools aggregate by entity_canonical_name when available.
    assert "entity_owner_canonical_name IF NOT EXISTS" in joined

    # Facts are relationships; fast tools frequently filter by predicate and need fact_id for evidence.
    assert "relates_owner_predicate IF NOT EXISTS" in joined
    assert "relates_fact_id IF NOT EXISTS" in joined
    assert "relates_owner_fact_id IF NOT EXISTS" in joined

    # Ingest stats are persisted per owner (no process-global state).
    assert "kg_ingest_meta_owner_unique IF NOT EXISTS" in joined

    # Alias/canonicalization layer (Phase 2): queryable canonical keys + aliases.
    assert "entity_canonical_id_unique IF NOT EXISTS" in joined
    assert "entity_alias_owner_text IF NOT EXISTS" in joined
