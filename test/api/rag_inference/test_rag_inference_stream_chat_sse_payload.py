import json
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from encapsulation.data_model.schema import Chunk

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


def test_rag_inference_stream_chat_sse_emits_payload_tool_call(monkeypatch, client):
    import api.routers.rag_inference as rag_router
    import api.routers.rag_inference_handlers as handlers

    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4())

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

    class FakeRAG:
        def stream_chat(self, _query, _owner_id, return_subgraph=False, progress_callback=None):  # noqa: ARG002
            if progress_callback:
                progress_callback({"stage": "retrieve", "status": "start"})
                progress_callback({"stage": "retrieve", "status": "end"})

            def _gen():
                yield "hello "
                yield "world"

            chunks = [Chunk(content="doc1", id="chunk-1")]
            subgraph_data = {"nodes": ["n1"]} if return_subgraph else None
            subgraph_info = {"stats": 1}
            return (_gen(), chunks, subgraph_data, subgraph_info)

    evidence_payload = {"chunks": [{"id": "chunk-1"}], "summary": "evidence ok"}
    session_handler = FakeSessionHandler()
    message_handler = FakeMessageHandler()
    rag = FakeRAG()
    monkeypatch.setattr(handlers, "_session_handler", session_handler)
    monkeypatch.setattr(handlers, "_message_handler", message_handler)
    monkeypatch.setattr(handlers, "_rag_inference_handler", rag)
    monkeypatch.setattr("api.routers.rag_inference_modules.stream_chat.utils.validators.validate_user_session", lambda *_: True)
    monkeypatch.setattr(
        "api.routers.rag_inference_modules.stream_chat.response.response_builder.build_chat_evidence",
        lambda *a, **k: evidence_payload,
    )

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello", "include_evidence": True, "return_subgraph": True},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200
    assert "data: [DONE]" in resp.text

    payload_args = None
    outer_subgraph = None
    content_parts: list[str] = []
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
        content_parts.append(delta.get("content") or "")
        for tool_call in delta.get("tool_calls") or []:
            fn = (tool_call or {}).get("function") or {}
            if fn.get("name") == "rag_arc_payload":
                payload_args = json.loads(fn.get("arguments") or "{}")
                outer_subgraph = chunk.get("subgraph")
                break

    assert "".join(content_parts) == "hello world"
    assert payload_args is not None
    assert payload_args["evidence"] == evidence_payload
    # Subgraph is carried at the outer chunk level (not inside the tool-call payload).
    assert outer_subgraph == {"nodes": ["n1"]}


def test_rag_inference_stream_chat_sse_renumbers_citations_and_sources(monkeypatch, client):
    import api.routers.rag_inference as rag_router
    import api.routers.rag_inference_handlers as handlers

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

    class FakeRAG:
        def stream_chat(self, _query, _owner_id, return_subgraph=False, progress_callback=None):  # noqa: ARG002
            if progress_callback:
                progress_callback({"stage": "retrieve", "status": "start"})
                progress_callback({"stage": "retrieve", "status": "end"})

            def _gen():
                yield "answer<sup>1</sup> mid<sup>3</sup> end"

            # Provide the same chunk ids that the evidence builder returns so the SSE
            # response builder can keep LLM/source keys stable.
            chunks = [Chunk(content=f"doc{i}", id=f"chunk-{i}") for i in range(1, 6)]
            subgraph_data = None
            subgraph_info = None
            return (_gen(), chunks, subgraph_data, subgraph_info)

    evidence_payload = {
        "chunks": [
            {"id": "chunk-1", "content": "doc1", "metadata": {"filename": "f1"}},
            {"id": "chunk-2", "content": "doc2", "metadata": {"filename": "f2"}},
            {"id": "chunk-3", "content": "doc3", "metadata": {"filename": "f3"}},
            {"id": "chunk-4", "content": "doc4", "metadata": {"filename": "f4"}},
            {"id": "chunk-5", "content": "doc5", "metadata": {"filename": "f5"}},
        ]
    }

    session_handler = FakeSessionHandler()
    message_handler = FakeMessageHandler()
    rag = FakeRAG()
    monkeypatch.setattr(handlers, "_session_handler", session_handler)
    monkeypatch.setattr(handlers, "_message_handler", message_handler)
    monkeypatch.setattr(handlers, "_rag_inference_handler", rag)
    monkeypatch.setattr("api.routers.rag_inference_modules.stream_chat.utils.validators.validate_user_session", lambda *_: True)
    monkeypatch.setattr(
        "api.routers.rag_inference_modules.stream_chat.response.response_builder.build_chat_evidence",
        lambda *a, **k: evidence_payload,
    )

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200

    payload_args = None
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
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") == "sources":
            sources_event = chunk
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        for tool_call in delta.get("tool_calls") or []:
            fn = (tool_call or {}).get("function") or {}
            if fn.get("name") == "rag_arc_payload":
                payload_args = json.loads(fn.get("arguments") or "{}")

    assert payload_args is not None
    assert sources_event is not None

    # The streamed tokens may contain non-contiguous citations, but the final payload stores the normalized answer.
    payload_text = ((payload_args.get("message") or {}).get("content") or {}).get("content")
    assert payload_text == "answer<sup>1</sup> mid<sup>2</sup> end"

    sources = sources_event.get("sources") or []
    assert [s.get("key") for s in sources] == [1, 2]
    assert [s.get("chunk_id") for s in sources] == ["chunk-1", "chunk-3"]
    assert sources_event.get("citation_key_map") == {"1": 1, "3": 2}
