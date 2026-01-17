import numpy as np

from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_retrieve import _PrunedHippoRAGNeo4jRetrieveMixin


class _DummyIndex:
    def __init__(self, *, ntotal: int, indices: list[int]):
        self.ntotal = ntotal
        self._indices = indices

    def search(self, vec: np.ndarray, k: int):  # noqa: ARG002
        # Return fixed indices regardless of query; distances are irrelevant for seeding.
        idxs = self._indices[:k]
        distances = np.zeros((1, len(idxs)), dtype=np.float32)
        indices = np.array([idxs], dtype=np.int64)
        return distances, indices


class _DummyEntityDB:
    def __init__(self):
        class _Cfg:
            metric = "cosine"
            normalize_L2 = True

        self.config = _Cfg()
        self.index = _DummyIndex(ntotal=3, indices=[0, 1, 2])
        self.index_to_docstore_id = {
            0: "entity-a",
            1: "chunk-x",
            2: "entity-b",
        }


class _DummyGraphStore:
    def __init__(self):
        self._db = _DummyEntityDB()
        # Only entity-b is present in the current node mapping (simulates owner-scoped graph).
        self.node_to_idx = {"entity-b": 1}

    def get_entity_faiss_db(self, owner_id):  # noqa: ARG002
        return self._db


class _DummyRetriever(_PrunedHippoRAGNeo4jRetrieveMixin):
    def __init__(self):
        self.graph_store = _DummyGraphStore()

    def _get_query_embedding(self, query: str) -> np.ndarray:  # noqa: ARG002
        return np.ones((4,), dtype=np.float32)


def test_seed_entities_from_entity_nn_filters_and_scopes():
    r = _DummyRetriever()
    seeds = r._seed_entities_from_entity_nn(query="q", owner_id="owner", top_k=10)  # type: ignore[arg-type]
    # chunk-x is filtered out; entity-a is filtered out by node_to_idx scoping.
    assert seeds == ["entity-b"]
