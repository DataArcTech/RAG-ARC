from types import SimpleNamespace

from core.retrieval.tantivy_bm25 import TantivyBM25Retriever


def _make_bm25_retriever():
    retriever = object.__new__(TantivyBM25Retriever)
    retriever.config = SimpleNamespace(search_kwargs={"k": 5, "with_score": False, "use_phrase_query": False})
    retriever._index = SimpleNamespace(index=None)
    return retriever


def test_bm25_requires_owner_id():
    retriever = _make_bm25_retriever()
    chunks = retriever._get_relevant_chunks("hello world", owner_id=None)
    assert chunks == []
