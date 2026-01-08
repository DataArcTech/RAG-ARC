import os
import uuid
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_client(*, user_type: int, shared_owner_id: uuid.UUID):
    import api.routers.chatbot as chatbot_router

    captured: dict[str, object] = {}

    class _DummyLLM:
        def stream_chat(self, messages):  # noqa: ANN001
            yield ""

    class _StubRAG:
        llm = _DummyLLM()

        def _build_messages_and_context(self, *, query, owner_id, return_subgraph):  # noqa: ANN001
            captured["owner_id"] = owner_id
            return ([{"role": "user", "content": query}], [], None, None)

    os.environ["CHATBOT_SHARED_DOCUMENT_OWNER_ID"] = str(shared_owner_id)
    user = SimpleNamespace(id=uuid.uuid4(), type=user_type)

    app = FastAPI()
    app.include_router(chatbot_router.router)
    app.dependency_overrides[chatbot_router.get_current_user] = lambda: user

    chatbot_router._reset_chatbot_modules_for_tests()
    chatbot_router._get_chatbot_rag_inference = lambda: _StubRAG()  # type: ignore[method-assign]

    client = TestClient(app)
    client._captured = captured  # type: ignore[attr-defined]
    client._user = user  # type: ignore[attr-defined]
    return client


def test_chatbot_uses_shared_owner_for_chatkb_users():
    shared_owner_id = uuid.uuid4()
    client = _build_client(user_type=1, shared_owner_id=shared_owner_id)
    payload = {
        "id": str(uuid.uuid4()),
        "content": "推荐点儿童保险",
        "messages": [{"role": "user", "content": "hi"}],  # avoid title generation path
        "stream": True,
    }
    resp = client.post("/api/messages", json=payload)
    assert resp.status_code == 200
    assert client._captured["owner_id"] == str(shared_owner_id)  # type: ignore[attr-defined]


def test_chatbot_uses_user_owner_for_livingkb_users():
    shared_owner_id = uuid.uuid4()
    client = _build_client(user_type=0, shared_owner_id=shared_owner_id)
    payload = {
        "id": str(uuid.uuid4()),
        "content": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = client.post("/api/messages", json=payload)
    assert resp.status_code == 200
    assert client._captured["owner_id"] == str(client._user.id)  # type: ignore[attr-defined]

