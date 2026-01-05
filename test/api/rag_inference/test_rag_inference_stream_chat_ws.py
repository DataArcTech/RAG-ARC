import json
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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


def test_rag_inference_stream_chat_ws_sends_payload(monkeypatch, client):
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
        async def chat_async(self, _history_text, owner_id, return_subgraph=False):  # noqa: ARG002
            assert owner_id == user.id
            return ("assistant ok", [], {"nodes": []} if return_subgraph else None, {"meta": 1})

    monkeypatch.setattr(rag_router, "session_handler", FakeSessionHandler())
    monkeypatch.setattr(rag_router, "message_handler", FakeMessageHandler())
    monkeypatch.setattr(rag_router, "rag_inference_handler", FakeRAG())
    monkeypatch.setattr(rag_router, "validate_user_session", lambda _session, _user: True)
    monkeypatch.setattr(rag_router, "build_chat_evidence", lambda *a, **k: {"chunks": [], "summary": "ok"})

    async def _ws_user(_websocket: WebSocket):  # noqa: ARG001
        return user

    client.app.dependency_overrides[rag_router.ws_get_current_user] = _ws_user
    try:
        with client.websocket_connect(f"/rag_inference/stream_chat/{session_id}") as ws:
            ws.send_text(json.dumps({"query": "hello", "include_evidence": True, "return_subgraph": True}))
            payload = ws.receive_json()
            assert payload["message"]["content"]["role"] == "assistant"
            assert payload["message"]["content"]["content"] == "assistant ok"
            assert "evidence" in payload
            assert payload["evidence"]["summary"] == "ok"
            assert "subgraph" in payload
            ws.close()
    finally:
        client.app.dependency_overrides.pop(rag_router.ws_get_current_user, None)


def test_rag_inference_stream_chat_ws_accepts_plain_text(monkeypatch, client):
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
        async def chat_async(self, history_text, owner_id, return_subgraph=False):  # noqa: ARG002
            assert "user: ping" in history_text
            return ("pong", [], None, None)

    monkeypatch.setattr(rag_router, "session_handler", FakeSessionHandler())
    monkeypatch.setattr(rag_router, "message_handler", FakeMessageHandler())
    monkeypatch.setattr(rag_router, "rag_inference_handler", FakeRAG())
    monkeypatch.setattr(rag_router, "validate_user_session", lambda _session, _user: True)

    async def _ws_user(_websocket: WebSocket):  # noqa: ARG001
        return user

    client.app.dependency_overrides[rag_router.ws_get_current_user] = _ws_user
    try:
        with client.websocket_connect(f"/rag_inference/stream_chat/{session_id}") as ws:
            ws.send_text("ping")
            payload = ws.receive_json()
            assert payload["message"]["content"]["content"] == "pong"
            ws.close()
    finally:
        client.app.dependency_overrides.pop(rag_router.ws_get_current_user, None)


def test_rag_inference_stream_chat_ws_rejects_include_all_owners_when_not_admin(monkeypatch, client):
    import api.routers.rag_inference as rag_router

    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4())

    class FakeSessionHandler:
        def get_session(self, _session_id):
            return SimpleNamespace(id=_session_id, user_id=user.id)

    monkeypatch.setattr(rag_router, "session_handler", FakeSessionHandler())
    monkeypatch.setattr(rag_router, "validate_user_session", lambda _session, _user: True)
    monkeypatch.setattr(rag_router, "is_admin_owner", lambda _owner_id: False)

    async def _ws_user(_websocket: WebSocket):  # noqa: ARG001
        return user

    client.app.dependency_overrides[rag_router.ws_get_current_user] = _ws_user
    try:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(f"/rag_inference/stream_chat/{session_id}") as ws:
                ws.send_text(json.dumps({"query": "hello", "include_all_owners": True}))
                ws.receive_json()
        assert excinfo.value.code == 1008
    finally:
        client.app.dependency_overrides.pop(rag_router.ws_get_current_user, None)
