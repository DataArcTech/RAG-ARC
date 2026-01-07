"""
测试 /rag_inference/stream_chat/{session_id} 接口的 owner_id 选择逻辑。

验证：
- type=0 (livingKB): 使用 current_user.id 作为 effective_owner
- type=1 (chatKB): 使用 _get_shared_document_owner_id() 作为 effective_owner
"""
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


def test_stream_chat_livingkb_uses_user_id_as_owner(monkeypatch, client):
    """测试 livingKB (type=0) 用户使用自己的 user_id 作为 owner_id"""
    import api.routers.rag_inference as rag_router

    session_id = uuid.uuid4()
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, type=0)  # livingKB user

    captured_owner_id = None

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
        def stream_chat(self, _history_text, owner_id, return_subgraph=False):  # noqa: ARG002
            nonlocal captured_owner_id
            captured_owner_id = owner_id

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
    # Mock _build_sources_for_frontend to avoid knowledge registration requirement
    monkeypatch.setattr(rag_router, "_build_sources_for_frontend", lambda _chunks, _max_sources: [])

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.post(
            f"/rag_inference/stream_chat/{session_id}",
            json={"query": "hello"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200
    # 验证 livingKB 用户使用自己的 user_id 作为 owner_id
    assert captured_owner_id == user_id


def test_stream_chat_chatkb_uses_shared_owner_id(monkeypatch, client):
    """测试 chatKB (type=1) 用户使用共享的 owner_id"""
    import api.routers.rag_inference as rag_router

    session_id = uuid.uuid4()
    user_id = uuid.uuid4()
    shared_owner_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, type=1)  # chatKB user

    captured_owner_id = None

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
        def stream_chat(self, _history_text, owner_id, return_subgraph=False):  # noqa: ARG002
            nonlocal captured_owner_id
            captured_owner_id = owner_id

            def _gen():
                yield "assistant ok"

            chunks = []
            subgraph_data = None
            subgraph_info = None
            return (_gen(), chunks, subgraph_data, subgraph_info)

    # Mock _get_shared_document_owner_id 返回共享的 owner_id
    monkeypatch.setattr(
        rag_router,
        "_get_shared_document_owner_id",
        lambda: shared_owner_id
    )
    monkeypatch.setattr(rag_router, "session_handler", FakeSessionHandler())
    monkeypatch.setattr(rag_router, "message_handler", FakeMessageHandler())
    monkeypatch.setattr(rag_router, "rag_inference_handler", FakeRAG())
    monkeypatch.setattr(rag_router, "validate_user_session", lambda _session, _user: True)
    # Mock _build_sources_for_frontend to avoid knowledge registration requirement
    monkeypatch.setattr(rag_router, "_build_sources_for_frontend", lambda _chunks, _max_sources: [])

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.post(
            f"/rag_inference/stream_chat/{session_id}",
            json={"query": "hello"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200
    # 验证 chatKB 用户使用共享的 owner_id，而不是自己的 user_id
    assert captured_owner_id == shared_owner_id
    assert captured_owner_id != user_id


def test_stream_chat_default_type_uses_user_id(monkeypatch, client):
    """测试没有 type 属性的用户（默认 type=0）使用自己的 user_id 作为 owner_id"""
    import api.routers.rag_inference as rag_router

    session_id = uuid.uuid4()
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id)  # 没有 type 属性，默认为 0

    captured_owner_id = None

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
        def stream_chat(self, _history_text, owner_id, return_subgraph=False):  # noqa: ARG002
            nonlocal captured_owner_id
            captured_owner_id = owner_id

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
    # Mock _build_sources_for_frontend to avoid knowledge registration requirement
    monkeypatch.setattr(rag_router, "_build_sources_for_frontend", lambda _chunks, _max_sources: [])

    client.app.dependency_overrides[rag_router.get_current_user] = lambda: user
    try:
        resp = client.post(
            f"/rag_inference/stream_chat/{session_id}",
            json={"query": "hello"},
        )
    finally:
        client.app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 200
    # 验证默认情况下使用自己的 user_id 作为 owner_id
    assert captured_owner_id == user_id

