import uuid

from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_retrieve import _PrunedHippoRAGNeo4jRetrieveMixin


class _StubStore:
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def get_neighbors_with_weights(self, node_id, owner_id=None):  # noqa: ARG002
        return list(self._neighbors.get(node_id, []))


class _Runner(_PrunedHippoRAGNeo4jRetrieveMixin):
    def __init__(self, neighbors):
        self.graph_store = _StubStore(neighbors)

    @staticmethod
    def _owner_to_str(owner_id):
        return str(owner_id) if owner_id is not None else None


def test_select_top_entity_chunks_prefers_top_entity_neighbor_chunks_then_falls_back():
    owner_id = uuid.uuid4()
    runner = _Runner(
        neighbors={
            "entity-a": [("chunk-1", 1.0), ("chunk-2", 0.1), ("entity-b", 2.0)],
        }
    )

    selected_ids, selected_scores, top_entity = runner._select_top_entity_chunks(
        ppr_scores_dict={
            "entity-a": 0.9,
            "entity-b": 0.1,
            "chunk-1": 0.2,
            "chunk-2": 0.8,
            "chunk-3": 0.7,
        },
        owner_id=owner_id,
        top_k=3,
        fallback_chunk_ids=["chunk-3", "chunk-2", "chunk-1"],
    )

    assert top_entity == "entity-a"
    # entity-a neighbors: chunk-2 (0.8) then chunk-1 (0.2), then fallback chunk-3 (0.7)
    assert selected_ids == ["chunk-2", "chunk-1", "chunk-3"]
    assert selected_scores == [0.8, 0.2, 0.7]

