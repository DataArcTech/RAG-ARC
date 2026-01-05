import json
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from encapsulation.data_model.schema import Chunk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


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
    monkeypatch.setattr(rag_router, "session_handler", FakeSessionHandler())
    monkeypatch.setattr(rag_router, "message_handler", FakeMessageHandler())
    monkeypatch.setattr(rag_router, "rag_inference_handler", FakeRAG())
    monkeypatch.setattr(rag_router, "validate_user_session", lambda _session, _user: True)
    monkeypatch.setattr(rag_router, "build_chat_evidence", lambda *a, **k: evidence_payload)

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
    content_parts: list[str] = []
    for raw_line in resp.text.splitlines():
        line = raw_line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        content_parts.append(delta.get("content") or "")
        for tool_call in delta.get("tool_calls") or []:
            fn = (tool_call or {}).get("function") or {}
            if fn.get("name") == "rag_arc_payload":
                payload_args = json.loads(fn.get("arguments") or "{}")
                break

    assert "".join(content_parts) == "hello world"
    assert payload_args is not None
    assert payload_args["evidence"] == evidence_payload
    assert payload_args["subgraph"] == {"nodes": ["n1"]}
