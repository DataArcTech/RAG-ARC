from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing import _PrunedHippoRAGNeo4jIndexingMixin


class _DummyStore(_PrunedHippoRAGNeo4jIndexingMixin):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def _execute_query(self, query: str, params=None):  # noqa: ANN001
        self.queries.append(query)
        if "MATCH (c:Chunk" in query and "MENTIONS" in query:
            return [
                {"entity_id": "e1", "entity_name": "Alice", "entity_type": "PERSON", "attributes": "{}"},
                {"entity_id": "e2", "entity_name": "Bob", "entity_type": "PERSON", "attributes": "{}"},
            ]
        return []


def test_get_graph_data_queries_relates_to_relationship() -> None:
    store = _DummyStore()
    store._get_graph_data("chunk_1")

    relation_queries = [q for q in store.queries if "MATCH (e1:Entity)" in q and "-[r:" in q]
    assert relation_queries, "expected _get_graph_data() to query entity->entity fact relationships"
    assert ":RELATES_TO" in relation_queries[0]
    assert ":Fact" not in relation_queries[0]

