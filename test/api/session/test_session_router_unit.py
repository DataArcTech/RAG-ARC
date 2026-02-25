import os
import sys
import uuid
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


@pytest.fixture
async def client(monkeypatch):
    import api.routers.session as session_router

    user = SimpleNamespace(id=uuid.uuid4())

    sessions: dict[uuid.UUID, SimpleNamespace] = {}
    messages: dict[uuid.UUID, list[dict]] = {}

    class _SessionMgr:
        def create_session(self, user_id, name):  # noqa: ARG002
            session_id = uuid.uuid4()
            sessions[session_id] = SimpleNamespace(id=session_id, user_id=user_id)
            messages[session_id] = []
            return session_id

        def list_sessions_by_user(self, user_id):
            return [{"id": str(s.id), "user_id": str(s.user_id)} for s in sessions.values() if s.user_id == user_id]

        def get_session(self, session_id):
            return sessions.get(session_id)

    class _MessageMgr:
        def create_message(self, message):  # noqa: ANN001
            payload = {
                "id": str(uuid.uuid4()),
                "session_id": str(message.session_id),
                "content": dict(message.content),
                "source_file_ids": None,
                "subgraph_data": None,
                "created_at": datetime.now().isoformat(),
            }
            messages[message.session_id].append(payload)
            return payload

        def list_messages_by_session(self, session_id):
            return []

    monkeypatch.setattr(session_router, "get_session_handler", lambda: _SessionMgr())
    monkeypatch.setattr(session_router, "get_message_handler", lambda: _MessageMgr())
    monkeypatch.setattr(session_router, "validate_user_session", lambda session, current_user: session.user_id == current_user.id)

    app = FastAPI()
    app.include_router(session_router.router)
    async def _stub_user():  # noqa: ANN001
        return user

    app.dependency_overrides[session_router.get_current_user] = _stub_user
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_session_create_list_and_post_message(client):
    create = await client.post("/session")
    assert create.status_code == 200
    session_id = create.json()

    sessions = await client.get("/session")
    assert sessions.status_code == 200
    assert any(s["id"] == session_id for s in sessions.json())

    msg = await client.post(f"/session/{session_id}/messages", json={"content": "hi"})
    assert msg.status_code == 200
    payload = msg.json()
    assert payload["content"]["role"] == "user"
    assert payload["content"]["content"] == "hi"


async def test_session_message_rejects_wrong_user(monkeypatch):
    import api.routers.session as session_router

    user = SimpleNamespace(id=uuid.uuid4())
    other = SimpleNamespace(id=uuid.uuid4())
    session_id = uuid.uuid4()

    class _SessionMgr:
        def get_session(self, _session_id):
            return SimpleNamespace(id=_session_id, user_id=other.id)

    monkeypatch.setattr(session_router, "get_session_handler", lambda: _SessionMgr())
    monkeypatch.setattr(session_router, "validate_user_session", lambda session, current_user: session.user_id == current_user.id)

    app = FastAPI()
    app.include_router(session_router.router)
    async def _stub_user():  # noqa: ANN001
        return user

    app.dependency_overrides[session_router.get_current_user] = _stub_user
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/session/{session_id}/messages", json={"content": "hi"})
    assert resp.status_code == 401
