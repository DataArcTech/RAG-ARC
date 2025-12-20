import threading
import uuid
from dataclasses import dataclass

from core.retrieval.tantivy_bm25 import TantivyBM25Retriever


class _StubSearchResult:
    def __init__(self):
        self.hits = []


class _StubSearcher:
    def search(self, _query, limit: int, order_by_field=None, order=None):  # noqa: ARG002
        return _StubSearchResult()

    def doc(self, doc_address: str):  # noqa: ARG002
        raise AssertionError("doc() should not be called when there are no hits")


class _StubTantivyIndex:
    def __init__(self):
        self.schema = object()

    def searcher(self):
        return _StubSearcher()


class _StubTokenizerManager:
    def get_current_tokenizer(self):
        return lambda _q: ["Enterprise", "Function"]

    def get_stopwords(self):
        return []


class _StubIndexBuilder:
    def __init__(self):
        self.index = _StubTantivyIndex()
        self.tokenizer_manager = _StubTokenizerManager()
        self.config = type("_C", (), {"index_path": ""})()

    def load_index(self, *_args, **_kwargs):
        return None


@dataclass
class _StubIndexConfig:
    def __init__(self, index_builder: _StubIndexBuilder):
        self._index_builder = index_builder

    def build(self):
        return self._index_builder


@dataclass
class _StubConfig:
    index_config: _StubIndexConfig
    search_kwargs: dict
    type: str = "stub_tantivy_bm25"


def test_tantivy_bm25_lowercases_query_tokens(monkeypatch):
    index_builder = _StubIndexBuilder()
    config = _StubConfig(index_config=_StubIndexConfig(index_builder=index_builder), search_kwargs={"k": 1})
    retriever = TantivyBM25Retriever(config)  # type: ignore[arg-type]

    captured: dict[str, list[str]] = {}

    def _capture(tokens, _use_phrase_query=False):  # noqa: ARG001
        captured["tokens"] = list(tokens)
        return object()

    monkeypatch.setattr(retriever, "_build_main_query", _capture)
    monkeypatch.setattr(retriever, "_build_filter_query", lambda *_args, **_kwargs: [])

    owner_id = str(uuid.uuid4())
    retriever.invoke("Enterprise Function", owner_id=owner_id, k=1)

    assert captured["tokens"] == ["enterprise", "function"]

