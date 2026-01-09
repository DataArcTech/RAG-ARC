from core.utils.rwlock import RWLock
from encapsulation.database.graph_db.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jStore


RAW_STORE_CLASS = getattr(PrunedHippoRAGNeo4jStore, "__wrapped__", PrunedHippoRAGNeo4jStore)


def test_graph_cache_load_adds_reverse_edges() -> None:
    store = object.__new__(RAW_STORE_CLASS)
    store._rwlock = RWLock()
    store._cache_loaded = False
    store._graph_cache = None
    store._entity_chunk_count_cache = None

    def _execute_query(_query: str, _params=None):  # noqa: ANN001
        return [
            {
                "node_id": "chunk-1",
                "neighbor_id": "entity-1",
                "relation_type": "MENTIONS",
                "weight": 2.0,
                "node_owner_id": RAW_STORE_CLASS.OWNER_GLOBAL_KEY,
                "neighbor_owner_id": RAW_STORE_CLASS.OWNER_GLOBAL_KEY,
                "relation_owner_id": RAW_STORE_CLASS.OWNER_GLOBAL_KEY,
            }
        ]

    store._execute_query = _execute_query
    store._load_graph_cache(force_reload=True)

    assert store.get_neighbors_with_weights("chunk-1") == [("entity-1", 2.0)]
    assert store.get_neighbors_with_weights("entity-1") == [("chunk-1", 2.0)]
    assert store.get_entity_chunk_count_from_cache("entity-1") == 1

