from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j


class _StubStore:
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    @staticmethod
    def _owner_key(owner_id):  # noqa: ANN001
        return str(owner_id)

    def _execute_query(self, cypher: str, params=None):  # noqa: ANN001
        self.calls.append((str(cypher or ""), dict(params or {})))
        # Minimal shape expected by exporter for count + node list.
        if "RETURN count(n) AS total_nodes" in cypher:
            return [{"total_nodes": 0}]
        if "RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id" in cypher:
            return []
        if "RETURN count(r)/2 AS total_edges" in cypher:
            return [{"total_edges": 0}]
        return []


def test_export_full_graph_enforces_owner_from_scope_label_when_owner_id_missing() -> None:
    store = _StubStore()
    GraphExporterNeo4j.export_full_graph(
        store,
        max_nodes=10,
        max_edges=10,
        include_node_types=["chunk", "entity"],
        owner_id=None,
        owner_scope_label="owner-1",
    )
    assert store.calls, "expected exporter to run cypher queries"
    # First query is the node count query and must include owner_id when scope label is present.
    cypher, params = store.calls[0]
    assert params.get("owner_id") == "owner-1"
    assert "WHERE (n:Chunk OR n:Entity)" in cypher


def test_export_subgraph_enforces_owner_from_scope_label_when_owner_id_missing() -> None:
    store = _StubStore()
    GraphExporterNeo4j.export_subgraph(
        store,
        subgraph_node_ids={"chunk-1"},
        owner_id=None,
        owner_scope_label="owner-2",
    )
    assert store.calls, "expected exporter to run cypher queries"
    # The first chunk node lookup should include owner_id.
    _, params = store.calls[0]
    assert params.get("owner_id") == "owner-2"
