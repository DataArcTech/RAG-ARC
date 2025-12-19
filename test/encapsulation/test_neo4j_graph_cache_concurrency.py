import threading

from encapsulation.database.graph_db.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jStore
from core.utils.rwlock import RWLock


RAW_STORE_CLASS = getattr(PrunedHippoRAGNeo4jStore, "__wrapped__", PrunedHippoRAGNeo4jStore)


def test_graph_cache_force_reload_does_not_expose_partial_state():
    store = object.__new__(RAW_STORE_CLASS)
    store._rwlock = RWLock()
    store._cache_loaded = True
    store._graph_cache = {RAW_STORE_CLASS.OWNER_GLOBAL_KEY: {"n1": [("n2", 1.0)]}}
    store._entity_chunk_count_cache = {}

    started = threading.Event()
    allow_query_return = threading.Event()

    def _execute_query(_query: str, _params=None):
        started.set()
        allow_query_return.wait(timeout=5)
        return [
            {
                "node_id": "n1",
                "neighbor_id": "n3",
                "weight": 2.0,
                "node_owner_id": RAW_STORE_CLASS.OWNER_GLOBAL_KEY,
                "neighbor_owner_id": RAW_STORE_CLASS.OWNER_GLOBAL_KEY,
                "relation_owner_id": RAW_STORE_CLASS.OWNER_GLOBAL_KEY,
            }
        ]

    store._execute_query = _execute_query

    thread = threading.Thread(target=lambda: store._load_graph_cache(force_reload=True))
    thread.start()
    assert started.wait(timeout=5)

    assert store.get_neighbors_with_weights("n1") == [("n2", 1.0)]

    allow_query_return.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert store.get_neighbors_with_weights("n1") == [("n3", 2.0)]

