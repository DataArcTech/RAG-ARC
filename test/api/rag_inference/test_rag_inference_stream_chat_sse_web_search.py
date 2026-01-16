import json
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from encapsulation.data_model.schema import Chunk
from encapsulation.web_search.tavily_client import TavilySearchClient, TavilySearchResult

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


def _seed_registry() -> None:
    from framework.register import Register

    reg = Register()
    reg.registrations.setdefault("account", object())
    reg.registrations.setdefault("chat_session", object())
    reg.registrations.setdefault("chat_message", object())
    reg.registrations.setdefault("rag_inference", object())


@pytest.fixture
def app():
    _seed_registry()
    import api.routers.rag_inference as rag_router

    app = FastAPI()
    app.include_router(rag_router.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_stream_chat_sse_includes_web_search_chunks_and_progress(monkeypatch, client):
    import api.routers.rag_inference as rag_router
    import api.routers.rag_inference_handlers as handlers
    from application.rag_inference.module import RAGInference

    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4(), type=0)

    class FakeSessionHandler:
        def get_session(self, _session_id):
            return SimpleNamespace(id=_session_id, user_id=user.id)

    class FakeMessageHandler:
        def __init__(self):
            self._messages = []

        def create_message(self, message):
            if getattr(message, "id", None) is None:
                setattr(message, "id", uuid.uuid4())
            self._messages.append(message)
            return message

        def list_messages_by_session(self, _session_id):
            return list(self._messages)

    class DummyQueryRewriter:
        def rewrite_query(self, q):  # noqa: ANN001
            return q

    class DummyRetriever:
        def invoke(self, q, **kwargs):  # noqa: ANN001
            k = int(kwargs.get("k") or 30)
            out = []
            for i in range(k):
                out.append(
                    Chunk(
                        id=f"chunk-{i}",
                        content=f"graph chunk {i} for {q}",
                        metadata={"score": float(k - i), "filename": "graph"},
                    )
                )
            return out

    class DummyReranker:
        def rerank(self, _query, chunks, **kwargs):  # noqa: ANN001
            top_k = kwargs.get("top_k")

            def _rank_key(ch):  # noqa: ANN001
                src = (getattr(ch, "metadata", None) or {}).get("source")
                return (0 if src == "web.tavily" else 1)

            ranked = sorted(chunks, key=_rank_key)
            if top_k is not None:
                return ranked[: int(top_k)]
            return ranked

        def get_reranker_info(self):
            return {"type": "dummy"}

    class DummyLLM:
        def stream_chat(self, _messages):  # noqa: ANN001
            def _gen():
                yield "assistant ok<sup>1</sup>"

            return _gen()

    class StubTavily:
        def search(self, *, query, max_results):  # noqa: ANN001
            results = []
            for i in range(int(max_results)):
                results.append(
                    TavilySearchResult(
                        title=f"Web {i}",
                        url=f"https://example.com/{i}",
                        content=f"snippet {i} for {query}",
                        score=1.0 - i * 0.1,
                    )
                )
            return results

        @staticmethod
        def to_evidence_chunks(*, results, step_id, query):  # noqa: ANN001
            return TavilySearchClient.to_evidence_chunks(results=results, step_id=step_id, query=query)

    rag = object.__new__(RAGInference)
    rag.config = SimpleNamespace(
        web_search=SimpleNamespace(timeout_seconds=1.0, timeout_grace_seconds=0.0, enabled=True),
        candidate_selection=SimpleNamespace(graph_candidates_k=30, web_candidates_k=5, rerank_keep_k=5),
    )
    rag.query_rewriter = DummyQueryRewriter()
    rag.retriever = DummyRetriever()
    rag.reranker = DummyReranker()
    rag.llm = DummyLLM()
    rag.graph_retriever = None
    rag._knowledge_module = None
    rag._tavily_client = StubTavily()

    session_handler = FakeSessionHandler()
    message_handler = FakeMessageHandler()
    monkeypatch.setattr(handlers, "_session_handler", session_handler)
    monkeypatch.setattr(handlers, "_message_handler", message_handler)
    monkeypatch.setattr(handlers, "_rag_inference_handler", rag)
    monkeypatch.setattr("api.routers.rag_inference_modules.stream_chat.validators.validate_user_session", lambda *_: True)

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello", "enable_web_search": "true"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200

    saw_web_search_start = False
    saw_web_search_end = False
    payload = None
    for raw_line in resp.text.splitlines():
        line = raw_line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        envelope = json.loads(data)
        chunk = envelope.get("data") if isinstance(envelope, dict) else None
        if not isinstance(chunk, dict):
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        for tool_call in delta.get("tool_calls") or []:
            fn = (tool_call or {}).get("function") or {}
            if fn.get("name") == "rag_arc_progress":
                progress = json.loads(fn.get("arguments") or "{}")
                if progress.get("stage") == "web_search" and progress.get("status") == "start":
                    saw_web_search_start = True
                if progress.get("stage") == "web_search" and progress.get("status") == "end":
                    saw_web_search_end = True
            if fn.get("name") == "rag_arc_payload":
                payload = json.loads(fn.get("arguments") or "{}")

    assert saw_web_search_start is True
    assert saw_web_search_end is True
    assert payload is not None
    assert "chunks" in payload and isinstance(payload["chunks"], list)
    assert len(payload["chunks"]) == 5
    # Payload chunk metadata is intentionally minimized; use content shape to detect web chunks.
    assert any(str(item.get("content") or "").startswith("Web ") for item in payload["chunks"])

    # sources event should include an external URL and keep UUID-like chunk_id for web sources
    sources_event = None
    for raw_line in resp.text.splitlines():
        line = raw_line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        envelope = json.loads(data)
        chunk = envelope.get("data") if isinstance(envelope, dict) else None
        if isinstance(chunk, dict) and chunk.get("type") == "sources":
            sources_event = chunk
            break
    assert sources_event is not None
    sources = sources_event.get("sources") or []
    assert isinstance(sources, list) and sources
    web_sources = [s for s in sources if isinstance(s, dict) and str(s.get("file") or "").startswith("http")]
    assert web_sources, "expected at least one web source with external URL"
    uuid.UUID(str(web_sources[0].get("chunk_id")))


def test_stream_chat_sse_defaults_to_no_web_search(monkeypatch, client):
    import api.routers.rag_inference as rag_router
    import api.routers.rag_inference_handlers as handlers
    from application.rag_inference.module import RAGInference

    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4(), type=0)

    class FakeSessionHandler:
        def get_session(self, _session_id):
            return SimpleNamespace(id=_session_id, user_id=user.id)

    class FakeMessageHandler:
        def __init__(self):
            self._messages = []

        def create_message(self, message):
            if getattr(message, "id", None) is None:
                setattr(message, "id", uuid.uuid4())
            self._messages.append(message)
            return message

        def list_messages_by_session(self, _session_id):
            return list(self._messages)

    class DummyQueryRewriter:
        def rewrite_query(self, q):  # noqa: ANN001
            return q

    class DummyRetriever:
        def invoke(self, q, **kwargs):  # noqa: ANN001
            k = int(kwargs.get("k") or 30)
            out = []
            for i in range(k):
                out.append(
                    Chunk(
                        id=f"chunk-{i}",
                        content=f"graph chunk {i} for {q}",
                        metadata={"score": float(k - i), "filename": "graph"},
                    )
                )
            return out

    class DummyReranker:
        def rerank(self, _query, chunks, **kwargs):  # noqa: ANN001
            top_k = kwargs.get("top_k")
            ranked = list(chunks)
            if top_k is not None:
                return ranked[: int(top_k)]
            return ranked

        def get_reranker_info(self):
            return {"type": "dummy"}

    class DummyLLM:
        def stream_chat(self, _messages):  # noqa: ANN001
            def _gen():
                yield "assistant ok"

            return _gen()

    class StubTavily:
        def search(self, *, query, max_results):  # noqa: ANN001
            results = []
            for i in range(int(max_results)):
                results.append(
                    TavilySearchResult(
                        title=f"Web {i}",
                        url=f"https://example.com/{i}",
                        content=f"snippet {i} for {query}",
                        score=1.0 - i * 0.1,
                    )
                )
            return results

        @staticmethod
        def to_evidence_chunks(*, results, step_id, query):  # noqa: ANN001
            return TavilySearchClient.to_evidence_chunks(results=results, step_id=step_id, query=query)

    rag = object.__new__(RAGInference)
    rag.config = SimpleNamespace(
        web_search=SimpleNamespace(timeout_seconds=1.0, enabled=True),
        candidate_selection=SimpleNamespace(graph_candidates_k=30, web_candidates_k=5, rerank_keep_k=5),
    )
    rag.query_rewriter = DummyQueryRewriter()
    rag.retriever = DummyRetriever()
    rag.reranker = DummyReranker()
    rag.llm = DummyLLM()
    rag.graph_retriever = None
    rag._knowledge_module = None
    rag._tavily_client = StubTavily()

    session_handler = FakeSessionHandler()
    message_handler = FakeMessageHandler()
    monkeypatch.setattr(handlers, "_session_handler", session_handler)
    monkeypatch.setattr(handlers, "_message_handler", message_handler)
    monkeypatch.setattr(handlers, "_rag_inference_handler", rag)
    monkeypatch.setattr("api.routers.rag_inference_modules.stream_chat.validators.validate_user_session", lambda *_: True)

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200

    saw_web_search_progress = False
    payload = None
    for raw_line in resp.text.splitlines():
        line = raw_line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        envelope = json.loads(data)
        chunk = envelope.get("data") if isinstance(envelope, dict) else None
        if not isinstance(chunk, dict):
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        for tool_call in delta.get("tool_calls") or []:
            fn = (tool_call or {}).get("function") or {}
            if fn.get("name") == "rag_arc_progress":
                progress = json.loads(fn.get("arguments") or "{}")
                if progress.get("stage") == "web_search":
                    saw_web_search_progress = True
            if fn.get("name") == "rag_arc_payload":
                payload = json.loads(fn.get("arguments") or "{}")

    assert saw_web_search_progress is False
    assert payload is not None
    chunks = payload.get("chunks") or []
    assert len(chunks) == 5
    assert not any(((item.get("metadata") or {}).get("source") == "web.tavily") for item in chunks)
