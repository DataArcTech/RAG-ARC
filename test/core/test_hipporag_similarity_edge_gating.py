from types import SimpleNamespace


class _StubGraphStore:
    def __init__(self, *, neighbors_by_node):
        self._neighbors_by_node = neighbors_by_node

    def get_neighbors_with_weights(self, node_id: str, owner_id=None):  # noqa: ANN001
        return [(nid, w) for nid, w, _rel in self._neighbors_by_node.get(node_id, [])]

    def get_batch_neighbors_with_weights(self, node_ids, owner_id=None):  # noqa: ANN001
        return {nid: [(n, w) for n, w, _rel in self._neighbors_by_node.get(nid, [])] for nid in node_ids}

    def get_batch_neighbors_with_weights_and_relations(self, node_ids, owner_id=None):  # noqa: ANN001
        return {nid: list(self._neighbors_by_node.get(nid, [])) for nid in node_ids}


def test_hipporag_neo4j_graph_expansion_gates_similarity_multi_hop():
    from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_graph import _PrunedHippoRAGNeo4jGraphMixin

    class _Runner(_PrunedHippoRAGNeo4jGraphMixin):
        def __init__(self):
            self.config = SimpleNamespace(
                expansion_hops=2,
                include_chunk_neighbors=True,
                max_neighbors=50,
                query_aware_multiplier=0.0,
                query_aware_min_k=1,
                query_aware_max_k=50,
                similarity_edge_relation="SIMILAR_TO",
                similarity_edge_max_hops=1,  # only hop0
                similarity_edge_min_similarity=0.8,
                similarity_edge_max_per_node=10,
            )
            self.passage_node_keys = ["chunk-1"]
            self.graph_store = _StubGraphStore(
                neighbors_by_node={
                    # seed entity
                    "entity-1": [
                        ("entity-2", 0.95, "SIMILAR_TO"),
                        ("chunk-1", 10.0, "MENTIONS"),
                    ],
                    # entity-2 connects to entity-3 only via SIMILAR_TO; should be blocked at hop1.
                    "entity-2": [
                        ("entity-3", 0.95, "SIMILAR_TO"),
                        ("chunk-1", 10.0, "MENTIONS"),
                    ],
                    "entity-3": [("chunk-1", 10.0, "MENTIONS")],
                }
            )

        @staticmethod
        def _owner_to_str(owner_id):  # noqa: ANN001
            return owner_id

    runner = _Runner()
    nodes, chunks = runner._expand_subgraph(seed_entity_ids={"entity-1"}, entity_relevance_scores=None, owner_id=None)
    assert "entity-2" in nodes
    assert "chunk-1" in nodes and "chunk-1" in chunks
    assert "entity-3" not in nodes
