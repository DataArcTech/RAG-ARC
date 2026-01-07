def test_neo4j_schema_statements_skip_existence_constraints_for_community():
    import pytest

    pytest.importorskip("igraph")
    from encapsulation.database.graph_db.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jStore

    stmts = PrunedHippoRAGNeo4jStore.__wrapped__.neo4j_schema_statements(include_existence_constraints=False)
    assert all("IS NOT NULL" not in stmt for stmt in stmts)
    assert any("IS UNIQUE" in stmt for stmt in stmts)

