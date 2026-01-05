from contextlib import nullcontext

from encapsulation.database.graph_db.pruned_hipporag_neo4j_cache import _PrunedHippoRAGNeo4jCacheMixin


class _CacheOnlyStore(_PrunedHippoRAGNeo4jCacheMixin):
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self):
        self._cache_loaded = True
        self._graph_cache = {}
        self._entity_chunk_count_cache = {}
        self._directed_fact_cache = {}
        self._directed_fact_cache_loaded_key = None

    def read_lock(self):
        return nullcontext()

    def write_lock(self):
        return nullcontext()

    def _owner_key(self, owner_id):  # noqa: ANN001
        return str(owner_id)

    def _execute_query(self, query, params=None):  # noqa: ANN001, ARG002
        raise AssertionError("This test should not hit Neo4j")


def _edge_weight(graph, src_idx: int, dst_idx: int) -> float | None:
    for edge in graph.es:
        u, v = edge.tuple
        if u == src_idx and v == dst_idx:
            return float(edge["weight"])
    return None


def test_directed_ppr_extraction_uses_cached_predicates_and_preserves_direction() -> None:
    store = _CacheOnlyStore()
    owner = "owner-1"

    # Undirected cache: only used for Chunk<->Entity edges in the directed extraction path.
    store._graph_cache[owner] = {
        "chunk-1": [("entity-a", 2.0)],
        "entity-a": [("chunk-1", 2.0)],
    }

    # Directed fact cache: Entity->Entity edges with normalized predicates.
    store._directed_fact_cache_loaded_key = owner
    store._directed_fact_cache[owner] = {
        "entity-a": [
            ("entity-b", 1.5, "OWNS"),
            ("entity-b", 1.0, "HAS_POLICY"),
        ]
    }

    graph, node_to_idx, _ = store.extract_subgraph_from_cache_for_ppr_directed(
        {"chunk-1", "entity-a", "entity-b"},
        owner_id=owner,
        directed_relations={"OWNS"},
    )
    assert graph.is_directed()

    idx_chunk = node_to_idx["chunk-1"]
    idx_a = node_to_idx["entity-a"]
    idx_b = node_to_idx["entity-b"]

    # Chunk<->Entity is undirected in directed igraph (both directions added).
    assert _edge_weight(graph, idx_chunk, idx_a) == 2.0
    assert _edge_weight(graph, idx_a, idx_chunk) == 2.0

    # OWNS is direction-sensitive: a->b is kept. HAS_POLICY is treated as undirected (both directions).
    assert _edge_weight(graph, idx_a, idx_b) == 2.5
    assert _edge_weight(graph, idx_b, idx_a) == 1.0

