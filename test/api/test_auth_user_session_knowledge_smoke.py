import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@pytest.fixture
def app(monkeypatch):
    import api.routers.auth as auth_router
    import api.routers.session as session_router
    import api.routers.user as user_router
    import api.routers.knowledge as knowledge_router

    class _StubAccount:
        def __init__(self):
            self._users: dict[str, dict] = {}

        async def register_user_async(self, user_create):  # noqa: ANN001
            payload = {
                "id": str(uuid.uuid4()),
                "name": user_create.name,
                "user_name": user_create.user_name,
            }
            self._users[user_create.user_name] = payload
            return payload

        async def authenticate_user_async(self, username: str, password: str):  # noqa: ARG002
            user = self._users.get(username)
            if user is None:
                return None
            return SimpleNamespace(**user)

        async def get_user_by_username_async(self, username: str):
            user = self._users.get(username)
            if user is None:
                return None
            return SimpleNamespace(**user)

    account = _StubAccount()

    monkeypatch.setattr(auth_router, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(auth_router, "ALGORITHM", "HS256")
    monkeypatch.setattr(auth_router, "get_account_handler", lambda: account)

    app = FastAPI()
    app.include_router(auth_router.router)
    app.include_router(user_router.router)
    app.include_router(session_router.router)
    app.include_router(knowledge_router.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _register_and_login(client: TestClient):
    username = f"u_{uuid.uuid4().hex[:8]}"
    password = "pw"
    resp = client.post("/auth/register", json={"name": "n", "user_name": username, "password": password})
    assert resp.status_code == 201
    token_resp = client.post(
        "/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]
    return username, token


def test_user_me_requires_auth(client):
    resp = client.get("/user/me")
    assert resp.status_code == 401


def test_auth_token_allows_user_me(client):
    _, token = _register_and_login(client)
    resp = client.get("/user/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["user_name"].startswith("u_")


def test_session_create_requires_auth(client):
    resp = client.post("/session")
    assert resp.status_code in {401, 422}


def test_knowledge_upload_requires_auth(client, tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello", encoding="utf-8")
    with file_path.open("rb") as f:
        resp = client.post("/knowledge", files={"file": ("a.txt", f, "text/plain")})
    assert resp.status_code == 401

