from core.retrieval.dense import DenseRetriever


class _DummyEmbedding:
    def embed(self, _text):  # noqa: ANN001
        # Deterministic stub; value doesn't matter for this routing test.
        return [0.0, 0.0, 0.0]


def test_owner_scoped_similarity_search_does_not_short_circuit_on__index_none() -> None:
    # DenseRetriever uses `self._index is None` as a normal "index not built" guard.
    # In owner-scoped mode, `self._index` is intentionally None and the retriever must
    # route through `similarity_search_by_vector()` which lazy-loads per-owner artifacts.
    r = DenseRetriever.__new__(DenseRetriever)
    r._owner_scoped_enabled = True
    r._index = None
    r.embedding = _DummyEmbedding()

    called: dict[str, object] = {}

    def _fake_ssbv(vec, include_score=False, **kwargs):  # noqa: ANN001
        called["ok"] = True
        assert vec == [0.0, 0.0, 0.0]
        assert include_score is False
        assert kwargs.get("owner_id") == "owner"
        return ["sentinel"]

    r.similarity_search_by_vector = _fake_ssbv  # type: ignore[method-assign]

    out = DenseRetriever.similarity_search(r, "query", include_score=False, owner_id="owner")
    assert out == ["sentinel"]
    assert called.get("ok") is True

