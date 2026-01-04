from contextlib import nullcontext

from encapsulation.database.graph_db.pruned_hipporag_neo4j_cache import _PrunedHippoRAGNeo4jCacheMixin


class _DummyCacheStore(_PrunedHippoRAGNeo4jCacheMixin):
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self):
        # Minimal caches required by the directed subgraph extractor.
        self._cache_loaded = True
        self._graph_cache = {
            "owner-a": {
                "chunk-1": [("entity-a", 1.0)],
                "entity-a": [("chunk-1", 1.0)],
            }
        }
        self._directed_fact_cache_loaded_key = "owner-a"
        self._directed_fact_cache = {
            "owner-a": {
                "entity-a": [
                    ("entity-b", 1.0, "OWNS"),
                    ("entity-c", 1.0, "RELATED_TO"),
                ]
            }
        }

    def read_lock(self):  # noqa: D401
        return nullcontext()

    def write_lock(self):  # noqa: D401
        return nullcontext()

    @classmethod
    def _owner_key(cls, owner_id):  # noqa: ANN001
        return str(owner_id)

    def _execute_query(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("Dummy cache store must not hit Neo4j")

    def _load_graph_cache(self, *_, **__):  # pragma: no cover
        raise AssertionError("Dummy cache store must not reload caches")


def _edge_set(graph, idx_to_node):
    edges = set()
    for src_idx, dst_idx in graph.get_edgelist():
        edges.add((idx_to_node[src_idx], idx_to_node[dst_idx]))
    return edges


def test_directed_subgraph_blacklist_policy_adds_reverse_only_for_undirected_relations():
    store = _DummyCacheStore()
    graph, _, idx_to_node = store.extract_subgraph_from_cache_for_ppr_directed(
        {"chunk-1", "entity-a", "entity-b", "entity-c"},
        owner_id="owner-a",
        direction_policy="blacklist",
        direction_insensitive_relations={"RELATED_TO"},
    )

    assert graph.is_directed()
    edges = _edge_set(graph, idx_to_node)

    # Chunk<->Entity always undirected (both directions).
    assert ("chunk-1", "entity-a") in edges
    assert ("entity-a", "chunk-1") in edges

    # OWNS should be direction-sensitive by default in blacklist mode.
    assert ("entity-a", "entity-b") in edges
    assert ("entity-b", "entity-a") not in edges

    # RELATED_TO is explicitly direction-insensitive: add reverse.
    assert ("entity-a", "entity-c") in edges
    assert ("entity-c", "entity-a") in edges


def test_directed_subgraph_whitelist_policy_adds_reverse_for_non_directed_relations():
    store = _DummyCacheStore()
    graph, _, idx_to_node = store.extract_subgraph_from_cache_for_ppr_directed(
        {"chunk-1", "entity-a", "entity-b", "entity-c"},
        owner_id="owner-a",
        direction_policy="whitelist",
        directed_relations={"OWNS"},
    )

    assert graph.is_directed()
    edges = _edge_set(graph, idx_to_node)

    assert ("entity-a", "entity-b") in edges
    assert ("entity-b", "entity-a") not in edges

    assert ("entity-a", "entity-c") in edges
    assert ("entity-c", "entity-a") in edges
