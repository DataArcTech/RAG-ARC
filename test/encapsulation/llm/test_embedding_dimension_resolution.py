import pytest

from encapsulation.llm.embedding.base import EmbeddingLLMBase


class _Cfg:
    def __init__(self, embedding_dimensions=None):
        self.embedding_dimensions = embedding_dimensions


class _DummyEmbedding(EmbeddingLLMBase):
    def __init__(self, config, vector_len: int = 7):
        super().__init__(config)
        self._vector_len = vector_len
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if isinstance(texts, str):
            return [0.0] * self._vector_len
        return [[0.0] * self._vector_len for _ in texts]

    async def aembed(self, texts):
        return self.embed(texts)

    def get_model_info(self):
        return {"model": "dummy"}


def test_get_embedding_dimension_prefers_configured_value():
    emb = _DummyEmbedding(_Cfg(embedding_dimensions=11), vector_len=7)
    assert emb.get_embedding_dimension() == 11
    assert emb.calls == 0


def test_get_embedding_dimension_autodetects_and_caches():
    emb = _DummyEmbedding(_Cfg(embedding_dimensions=None), vector_len=9)
    assert emb.get_embedding_dimension() == 9
    assert emb.calls == 1
    assert emb.get_embedding_dimension() == 9
    assert emb.calls == 1

