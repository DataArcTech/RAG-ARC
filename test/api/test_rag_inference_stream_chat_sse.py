import json
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from main import app


@pytest.fixture
def client():
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
        def chat(self, _history_text, _owner_id, return_subgraph=False):
            chunks = []
            subgraph_data = None
            subgraph_info = None
            return ("assistant ok", chunks, subgraph_data, subgraph_info)

    monkeypatch.setattr(rag_router, "session_handler", FakeSessionHandler())
    monkeypatch.setattr(rag_router, "message_handler", FakeMessageHandler())
    monkeypatch.setattr(rag_router, "rag_inference_handler", FakeRAG())
    monkeypatch.setattr(rag_router, "validate_user_session", lambda _session, _user: True)

    app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        )
    finally:
        app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200
    assert "event: message" in resp.text

    payload = None
    current_event = None
    for raw_line in resp.text.splitlines():
        line = raw_line.strip("\r")
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:") and current_event == "message":
            payload = json.loads(line.split(":", 1)[1].strip())
            break

    assert payload is not None
    assert payload["message"]["content"]["content"] == "assistant ok"


def test_rag_inference_stream_chat_sse_requires_auth(monkeypatch, client):
    import api.routers.rag_inference as rag_router

    session_id = uuid.uuid4()

    app.dependency_overrides[rag_router.get_current_user] = lambda: None
    try:
        resp = client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        )
    finally:
        app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 401
