import os
from types import SimpleNamespace

import pytest


class DummyEmbedding:
    def __init__(self, dim: int):
        self._dim = dim
        self.set_calls: list[int] = []

    def embed(self, _text):
        return [0.0] * self._dim

    def set_embedding_dimensions(self, dim: int) -> bool:
        self.set_calls.append(int(dim))
        self._dim = int(dim)
        return True


def test_dense_retriever_aligns_embedding_dimension(monkeypatch: pytest.MonkeyPatch):
    from core.retrieval.dense import DenseRetriever

    monkeypatch.setenv("AUTO_ALIGN_EMBEDDING_DIM", "1")

    dr = DenseRetriever.__new__(DenseRetriever)
    dr.embedding = DummyEmbedding(dim=1536)
    dr._index = SimpleNamespace(index=SimpleNamespace(d=384))
    dr.config = SimpleNamespace(index_config=SimpleNamespace(embedding_config=SimpleNamespace(embedding_dimensions=None)))

    out = DenseRetriever._embed_query_aligned(dr, "hello")
    assert isinstance(out, list)
    assert len(out) == 384
    assert dr.embedding.set_calls == [384]


def test_dense_retriever_respects_explicit_embedding_dimensions(monkeypatch: pytest.MonkeyPatch):
    from core.retrieval.dense import DenseRetriever

    monkeypatch.setenv("AUTO_ALIGN_EMBEDDING_DIM", "1")

    dr = DenseRetriever.__new__(DenseRetriever)
    dr.embedding = DummyEmbedding(dim=1536)
    dr.embedding.supports_dimension_override = lambda: True
    dr._index = SimpleNamespace(index=SimpleNamespace(d=384))
    dr.config = SimpleNamespace(index_config=SimpleNamespace(embedding_config=SimpleNamespace(embedding_dimensions=1536)))

    with pytest.raises(RuntimeError):
        DenseRetriever._embed_query_aligned(dr, "hello")


def test_pruned_hipporag_get_query_embedding_aligns(monkeypatch: pytest.MonkeyPatch):
    from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever

    monkeypatch.setenv("AUTO_ALIGN_EMBEDDING_DIM", "1")

    retriever = PrunedHippoRAGRetriever.__new__(PrunedHippoRAGRetriever)
    retriever.embedding_model = DummyEmbedding(dim=1536)
    retriever.config = SimpleNamespace(graph_config=SimpleNamespace(embedding=SimpleNamespace(embedding_dimensions=None)))

    emb = PrunedHippoRAGRetriever._get_query_embedding(retriever, "q", expected_dim=384)
    assert int(getattr(emb, "shape", [0])[0]) == 384


def test_pruned_hipporag_get_query_embedding_raises_when_cannot_align(monkeypatch: pytest.MonkeyPatch):
    from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever

    class NoAlignEmbedding:
        def embed(self, _text):
            return [0.0] * 1536

    monkeypatch.setenv("AUTO_ALIGN_EMBEDDING_DIM", "1")

    retriever = PrunedHippoRAGRetriever.__new__(PrunedHippoRAGRetriever)
    retriever.embedding_model = NoAlignEmbedding()
    retriever.config = SimpleNamespace(graph_config=SimpleNamespace(embedding=SimpleNamespace(embedding_dimensions=None)))

    with pytest.raises(RuntimeError):
        PrunedHippoRAGRetriever._get_query_embedding(retriever, "q", expected_dim=384)
