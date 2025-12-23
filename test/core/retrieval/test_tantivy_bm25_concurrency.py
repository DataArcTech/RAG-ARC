import os
import sys
import uuid
import threading
from dataclasses import dataclass


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.retrieval.tantivy_bm25 import TantivyBM25Retriever


class _StubDoc:
    def __init__(self, owner_id: str, doc_id: str):
        self._owner_id = owner_id
        self._doc_id = doc_id

    def get_first(self, key: str):
        if key == "owner_id":
            return self._owner_id
        if key == "id":
            return self._doc_id
        if key == "content":
            return f"content for {self._doc_id}"
        if key == "metadata":
            return {}
        return None


class _StubSearcher:
    def __init__(self, owner_id: str, barrier: threading.Barrier):
        self._owner_id = owner_id
        self._barrier = barrier

    def search(self, _query, limit: int, order_by_field=None, order=None):
        self._barrier.wait(timeout=5)
        hits = [(1.0, f"{self._owner_id}:0")]

        class _Result:
            def __init__(self, hits):
                self.hits = hits

        return _Result(hits[:limit])

    def doc(self, doc_address: str):
        owner_prefix = doc_address.split(":", 1)[0]
        assert owner_prefix == self._owner_id, "doc() called on the wrong searcher instance"
        return _StubDoc(owner_id=self._owner_id, doc_id=doc_address)


class _StubTantivyIndex:
    def __init__(self, owner_by_thread: dict[int, str], barrier: threading.Barrier):
        self._owner_by_thread = owner_by_thread
        self._barrier = barrier
        self.schema = object()

    def searcher(self):
        owner_id = self._owner_by_thread[threading.get_ident()]
        return _StubSearcher(owner_id=owner_id, barrier=self._barrier)


class _StubTokenizerManager:
    def get_current_tokenizer(self):
        return lambda _q: ["stub"]


class _StubIndexBuilder:
    def __init__(self, owner_by_thread: dict[int, str], barrier: threading.Barrier):
        self.index = _StubTantivyIndex(owner_by_thread=owner_by_thread, barrier=barrier)
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


def test_tantivy_bm25_retriever_concurrent_searcher_isolation(monkeypatch):
    barrier = threading.Barrier(2)
    owner_by_thread: dict[int, str] = {}
    index_builder = _StubIndexBuilder(owner_by_thread=owner_by_thread, barrier=barrier)
    config = _StubConfig(index_config=_StubIndexConfig(index_builder=index_builder), search_kwargs={"k": 1})

    retriever = TantivyBM25Retriever(config)  # type: ignore[arg-type]

    monkeypatch.setattr(retriever, "_build_main_query", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(retriever, "_build_filter_query", lambda *_args, **_kwargs: [])

    owner_a = str(uuid.uuid4())
    owner_b = str(uuid.uuid4())
    results: dict[str, list[str]] = {}

    def _worker(owner_id: str):
        owner_by_thread[threading.get_ident()] = owner_id
        chunks = retriever.invoke("q", owner_id=owner_id, k=1)
        results[owner_id] = [c.id for c in chunks]

    thread_a = threading.Thread(target=_worker, args=(owner_a,))
    thread_b = threading.Thread(target=_worker, args=(owner_b,))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()

    assert results[owner_a] == [f"{owner_a}:0"]
    assert results[owner_b] == [f"{owner_b}:0"]

