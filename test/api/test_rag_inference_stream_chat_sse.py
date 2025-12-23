import json
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def _seed_registry() -> None:
    """Ensure router imports do not require real DB-backed registrations."""

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


def test_rag_inference_stream_chat_sse_emits_message_event(monkeypatch, client):
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
        def stream_chat(self, _history_text, _owner_id, return_subgraph=False):  # noqa: ARG002
            def _gen():
                yield "assistant ok"

            chunks = []
            subgraph_data = None
            subgraph_info = None
            return (_gen(), chunks, subgraph_data, subgraph_info)

    monkeypatch.setattr(rag_router, "session_handler", FakeSessionHandler())
    monkeypatch.setattr(rag_router, "message_handler", FakeMessageHandler())
    monkeypatch.setattr(rag_router, "rag_inference_handler", FakeRAG())
    monkeypatch.setattr(rag_router, "validate_user_session", lambda _session, _user: True)

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200
    assert '"object":"chat.completion.chunk"' in resp.text
    assert "data: [DONE]" in resp.text
    assert '"name":"rag_arc_progress"' in resp.text

    parts: list[str] = []
    first_chunk = None
    progress_payload = None
    for raw_line in resp.text.splitlines():
        line = raw_line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if first_chunk is None:
            first_chunk = chunk
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        for tool_call in delta.get("tool_calls") or []:
            fn = (tool_call or {}).get("function") or {}
            if fn.get("name") == "rag_arc_progress" and progress_payload is None:
                progress_payload = json.loads(fn.get("arguments") or "{}")
        parts.append(delta.get("content") or "")

    assert first_chunk is not None
    first_delta = (first_chunk.get("choices") or [{}])[0].get("delta") or {}
    assert first_delta.get("role") == "assistant"
    assert progress_payload is not None
    assert progress_payload.get("v") == 1
    assert progress_payload.get("type") == "progress"
    assert progress_payload.get("stage") == "prepare"
    assert isinstance(progress_payload.get("request_id"), str) and progress_payload.get("request_id")
    assert isinstance(progress_payload.get("seq"), int) and progress_payload.get("seq") >= 1
    assert "".join(parts) == "assistant ok"


def test_rag_inference_stream_chat_sse_requires_auth(monkeypatch, client):
    import api.routers.rag_inference as rag_router

    session_id = uuid.uuid4()

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: None
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 401
