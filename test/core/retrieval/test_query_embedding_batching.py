import numpy as np

from core.retrieval.graph_retrieveal.pruned_hipporag_facts import _PrunedHippoRAGFactsMixin


class _StubEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def embed(self, texts):
        self.calls.append(texts)
        if isinstance(texts, list):
            return [[1.0, 2.0, 3.0] for _ in texts]
        return [1.0, 2.0, 3.0]


class _Dummy(_PrunedHippoRAGFactsMixin):
    pass


def test_get_query_embeddings_batches_single_remote_call():
    dummy = _Dummy()
    dummy.embedding_model = _StubEmbeddingModel()

    out = dummy._get_query_embeddings(["q1", "q2"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert len(dummy.embedding_model.calls) == 1
    assert dummy.embedding_model.calls[0] == ["q1", "q2"]

