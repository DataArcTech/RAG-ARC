import asyncio
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _ensure_env_for_chatbot_tests() -> None:
    os.environ["RAG_INFERENCE_CONFIG_PATH"] = "config/json_configs/chatbot_test/rag_inference.json"
    os.environ["KNOWLEDGE_CONFIG_PATH"] = "config/json_configs/chatbot_test/knowledge.json"
    os.environ["CHATBOT_RAG_INFERENCE_CONFIG_PATH"] = "config/json_configs/chatbot_test/rag_inference.json"
    os.environ["CHATBOT_KNOWLEDGE_CONFIG_PATH"] = "config/json_configs/chatbot_test/knowledge.json"
    os.environ["CHATBOT_MAX_CONCURRENCY"] = "16"
    os.environ["CHATBOT_MAX_CONTEXT_TOKENS"] = "512"
    os.environ["CHATBOT_STREAM_CHUNK_CHARS"] = "16"
    os.environ["CHATBOT_CONTEXT_TURNS"] = "5"
    os.environ["LOCAL_FILE_STORAGE_PATH"] = "./test_output/chatbot_local_files"


def _cleanup_test_output() -> None:
    roots = [
        Path("test_output/chatbot_file_store"),
        Path("test_output/chatbot_parsed_content_store"),
        Path("test_output/chatbot_chunk_store"),
        Path("test_output/chatbot_parsed_files"),
        Path("test_output/chatbot_bm25_index"),
        Path("test_output/chatbot_local_files"),
    ]
    for root in roots:
        if root.exists():
            shutil.rmtree(root)
    Path("test_output").mkdir(parents=True, exist_ok=True)
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)


def _ensure_user_in_db(user_id: uuid.UUID) -> None:
    from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
    from encapsulation.data_model.orm_models import User

    db = PostgreSQLConfig().build()
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    with db.SessionMaker() as session:
        existing = session.query(User).filter_by(id=user_id).first()
        if existing is None:
            session.add(
                User(
                    id=user_id,
                    user_name=f"chatbot_test_{str(user_id)[:8]}",
                    hashed_password="dummy_hash",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()


def _ingest_doc_for_owner(owner_id: uuid.UUID, filename: str, content: str) -> str:
    from api.routers import chatbot as chatbot_router

    knowledge = chatbot_router._get_chatbot_knowledge()
    file_id = knowledge.file_storage.upload_file(
        filename=filename,
        file_data=content.encode("utf-8"),
        owner_id=owner_id,
        content_type="text/plain",
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("_ingest_doc_for_owner cannot be called from an async test; use _aingest_doc_for_owner")
    result = asyncio.run(knowledge.file_index.index_file(file_id))
    assert result.get("success") is True, result
    return file_id


async def _aingest_doc_for_owner(owner_id: uuid.UUID, filename: str, content: str) -> str:
    from api.routers import chatbot as chatbot_router

    knowledge = chatbot_router._get_chatbot_knowledge()
    file_id = await asyncio.to_thread(
        knowledge.file_storage.upload_file,
        filename,
        content.encode("utf-8"),
        owner_id,
        "text/plain",
    )
    result = await knowledge.file_index.index_file(file_id)
    assert result.get("success") is True, result
    return file_id


@pytest.fixture(scope="module")
def app_client():
    _ensure_env_for_chatbot_tests()
    _cleanup_test_output()

    from api.routers import chatbot as chatbot_router

    chatbot_router._reset_chatbot_modules_for_tests()

    from main import app

    return TestClient(app)


def test_chatbot_conversation_and_evidence(app_client: TestClient):
    r = app_client.get("/chatbot/bootstrap")
    assert r.status_code == 200
    browser_user_id = uuid.UUID(r.json()["browser_user_id"])

    _ensure_user_in_db(browser_user_id)
    admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
    _ensure_user_in_db(admin_id)
    file_id = _ingest_doc_for_owner(
        admin_id,
        filename="facts.txt",
        content="Paris is the capital of France.\nRAG-ARC is a retrieval augmented generation project.\n",
    )

    conversation_id = str(uuid.uuid4())
    payload = {
        "conversation_id": conversation_id,
        "message": {"role": "user", "content": "What is the capital of France?"},
        "memory": {"version": 0, "summary": "", "recent_messages": []},
        "options": {"include_evidence": True, "top_k": 3, "return_subgraph": False, "max_context_fraction": 0.5},
    }
    resp = app_client.post("/chatbot/chat", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert uuid.UUID(data["request_id"])
    assert data["browser_user_id"] == str(browser_user_id)
    assert data["conversation_id"] == conversation_id
    assert data["assistant"]["role"] == "assistant"
    assert data["assistant"]["content"]
    assert "Sources: [1]" in data["assistant"]["content"]
    assert data["memory"]["version"] >= 1
    assert isinstance(data["citations"], list)
    assert len(data["citations"]) >= 1

    citation = data["citations"][0]
    assert citation["chunk_url"].startswith("/chatbot/chunks/")
    assert citation["file_url"].startswith("/chatbot/files/")
    assert citation["file_id"] == file_id
    assert citation["filename"] == "facts.txt"
    assert citation["preview"]

    chunk_resp = app_client.get(citation["chunk_url"])
    assert chunk_resp.status_code == 200
    chunk_data = chunk_resp.json()
    assert chunk_data["file_id"] == file_id
    assert chunk_data["filename"] == "facts.txt"
    assert "Paris is the capital of France" in chunk_data["content"]

    file_resp = app_client.get(citation["file_url"])
    assert file_resp.status_code == 200
    assert "inline" in file_resp.headers.get("Content-Disposition", "")
    assert b"Paris is the capital of France" in file_resp.content

    title_resp = app_client.post(
        "/chatbot/title",
        json={
            "conversation_id": conversation_id,
            "user": payload["message"]["content"],
            "assistant": data["assistant"]["content"],
        },
    )
    assert title_resp.status_code == 200, title_resp.text
    title_data = title_resp.json()
    assert title_data["conversation_id"] == conversation_id
    assert title_data["browser_user_id"] == str(browser_user_id)
    assert title_data["title"]


def test_chatbot_multi_round_and_history_is_local(app_client: TestClient):
    r = app_client.get("/chatbot/bootstrap")
    assert r.status_code == 200
    browser_user_id = uuid.UUID(r.json()["browser_user_id"])
    _ensure_user_in_db(browser_user_id)
    admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
    _ensure_user_in_db(admin_id)
    _ingest_doc_for_owner(
        admin_id,
        filename="project.txt",
        content="alpha alpha alpha\nThis doc is for multi-round testing.\n",
    )

    conversation_id = str(uuid.uuid4())
    memory = {"version": 0, "summary": "", "recent_messages": []}

    first = app_client.post(
        "/chatbot/chat",
        json={
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": "alpha"},
            "memory": memory,
            "options": {"include_evidence": True, "top_k": 2, "return_subgraph": False, "max_context_fraction": 0.9},
        },
    )
    assert first.status_code == 200, first.text
    first_data = first.json()
    memory = first_data["memory"]
    assert len(memory["recent_messages"]) >= 2
    assert "Sources: [1]" in first_data["assistant"]["content"]

    second = app_client.post(
        "/chatbot/chat",
        json={
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": "Repeat alpha in one sentence."},
            "memory": memory,
            "options": {"include_evidence": True, "top_k": 2, "return_subgraph": False, "max_context_fraction": 0.9},
        },
    )
    assert second.status_code == 200, second.text
    second_data = second.json()
    assert second_data["memory"]["version"] > first_data["memory"]["version"]
    assert "Sources: [1]" in second_data["assistant"]["content"]

    base = Path("local/tmp/chatbot_history_test") / str(browser_user_id)
    base.mkdir(parents=True, exist_ok=True)
    history_path = base / f"{conversation_id}.json"
    history_path.write_text(
        '{"conversation_id": "%s", "turns": %d}' % (conversation_id, 2),
        encoding="utf-8",
    )
    assert history_path.exists()


def test_chatbot_cookie_owner_isolation_local_history(app_client: TestClient):
    client_a = app_client
    client_b = TestClient(client_a.app)

    a_boot = client_a.get("/chatbot/bootstrap")
    b_boot = client_b.get("/chatbot/bootstrap")
    assert a_boot.status_code == 200
    assert b_boot.status_code == 200

    a_uid = uuid.UUID(a_boot.json()["browser_user_id"])
    b_uid = uuid.UUID(b_boot.json()["browser_user_id"])
    assert a_uid != b_uid

    a_dir = Path("local/tmp/chatbot_history_test") / str(a_uid)
    b_dir = Path("local/tmp/chatbot_history_test") / str(b_uid)
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    (a_dir / "history.json").write_text('{"owner": "a"}', encoding="utf-8")
    (b_dir / "history.json").write_text('{"owner": "b"}', encoding="utf-8")

    assert (a_dir / "history.json").read_text(encoding="utf-8") != (b_dir / "history.json").read_text(encoding="utf-8")


def test_chatbot_chunk_and_file_access_is_shared(app_client: TestClient):
    client_a = app_client
    client_b = TestClient(client_a.app)
    client_anon = TestClient(client_a.app)

    a_boot = client_a.get("/chatbot/bootstrap")
    b_boot = client_b.get("/chatbot/bootstrap")
    assert a_boot.status_code == 200
    assert b_boot.status_code == 200
    a_uid = uuid.UUID(a_boot.json()["browser_user_id"])
    b_uid = uuid.UUID(b_boot.json()["browser_user_id"])
    _ensure_user_in_db(a_uid)
    _ensure_user_in_db(b_uid)

    admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
    _ensure_user_in_db(admin_id)
    file_id = _ingest_doc_for_owner(admin_id, filename="a.txt", content="alphaA alphaA alphaA\n")
    conversation_id = str(uuid.uuid4())
    resp = client_a.post(
        "/chatbot/chat",
        json={
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": "alphaA"},
            "memory": {"version": 0, "summary": "", "recent_messages": []},
            "options": {"include_evidence": True, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.5},
        },
    )
    assert resp.status_code == 200, resp.text
    resp_data = resp.json()
    assert "Sources: [1]" in resp_data["assistant"]["content"]
    cite = resp_data["citations"][0]
    assert cite["file_id"] == file_id

    # Different cookie -> still allowed (shared document library)
    assert client_b.get(cite["chunk_url"]).status_code == 200
    assert client_b.get(cite["file_url"]).status_code == 200

    # No cookie at all -> unauthorized
    assert client_anon.get(cite["chunk_url"]).status_code == 401
    assert client_anon.get(cite["file_url"]).status_code == 401


def test_chatbot_streaming_ws(app_client: TestClient):
    boot = app_client.get("/chatbot/bootstrap")
    assert boot.status_code == 200
    conversation_id = str(uuid.uuid4())

    with app_client.websocket_connect(f"/chatbot/ws?conversation_id={conversation_id}") as ws:
        ws.send_json(
            {
                "message": {"role": "user", "content": "Stream this reply"},
                "memory": {"version": 0, "summary": "", "recent_messages": []},
                "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
            }
        )
        start = ws.receive_json()
        assert start["type"] == "start"
        request_id = start["request_id"]

        deltas = []
        final = None
        for _ in range(100):
            msg = ws.receive_json()
            if msg["type"] == "delta":
                assert msg["request_id"] == request_id
                deltas.append(msg["content"])
            if msg["type"] == "final":
                final = msg
                break
        assert final is not None
        streamed = "".join(deltas)
        assert streamed
        assert final["assistant"]["content"].startswith(streamed[:8])

        title_msg = ws.receive_json()
        assert title_msg["type"] == "title"
        assert title_msg["request_id"] == request_id
        assert title_msg["conversation_id"] == conversation_id
        assert title_msg["title"]


def test_chatbot_owner_id_header_works_without_cookie(app_client: TestClient):
    owner = uuid.uuid4()
    _ensure_user_in_db(owner)

    conversation_id = str(uuid.uuid4())
    resp = app_client.post(
        "/chatbot/chat",
        headers={"X-Owner-Id": str(owner)},
        json={
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": "hello"},
            "memory": {"version": 0, "summary": "", "recent_messages": []},
            "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["browser_user_id"] == str(owner)


def test_chatbot_last_5_turns_window_allows_long_conversation(app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHATBOT_MAX_CONTEXT_TOKENS", "512")
    monkeypatch.setenv("CHATBOT_CONTEXT_TURNS", "5")

    boot = app_client.get("/chatbot/bootstrap")
    assert boot.status_code == 200
    uid = uuid.UUID(boot.json()["browser_user_id"])
    _ensure_user_in_db(uid)

    conversation_id = str(uuid.uuid4())
    memory = {"version": 0, "summary": "", "recent_messages": []}
    for i in range(1, 16):
        r = app_client.post(
            "/chatbot/chat",
            json={
                "conversation_id": conversation_id,
                "message": {"role": "user", "content": f"{i}:" + ("m" * 80)},
                "memory": memory,
                "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        memory = data["memory"]
        assert len(memory.get("recent_messages") or []) <= 10

    assert memory.get("recent_messages")

@pytest.mark.anyio
async def test_chatbot_multi_user_concurrency():
    _ensure_env_for_chatbot_tests()
    from api.routers import chatbot as chatbot_router

    chatbot_router._reset_chatbot_modules_for_tests()
    _cleanup_test_output()

    from main import app

    transport1 = httpx.ASGITransport(app=app)
    transport2 = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport1, base_url="http://test") as c1, httpx.AsyncClient(
        transport=transport2, base_url="http://test"
    ) as c2:
        b1 = await c1.get("/chatbot/bootstrap")
        b2 = await c2.get("/chatbot/bootstrap")
        u1 = uuid.UUID(b1.json()["browser_user_id"])
        u2 = uuid.UUID(b2.json()["browser_user_id"])
        _ensure_user_in_db(u1)
        _ensure_user_in_db(u2)
        admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
        _ensure_user_in_db(admin_id)
        await _aingest_doc_for_owner(admin_id, "u1.txt", "User one content about concurrency.\n")
        await _aingest_doc_for_owner(admin_id, "u2.txt", "User two content about concurrency.\n")

        async def ask(client: httpx.AsyncClient, uid: uuid.UUID) -> httpx.Response:
            return await client.post(
                "/chatbot/chat",
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "message": {"role": "user", "content": f"Hello from {uid}"},
                    "memory": {"version": 0, "summary": "", "recent_messages": []},
                    "options": {"include_evidence": True, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
                },
            )

        started = time.monotonic()
        r1, r2 = await asyncio.gather(ask(c1, u1), ask(c2, u2))
        elapsed = time.monotonic() - started

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert elapsed < 0.7, f"expected concurrency, took {elapsed:.2f}s"


@pytest.mark.anyio
async def test_chatbot_context_too_long_does_not_block_other_requests():
    _ensure_env_for_chatbot_tests()
    os.environ["CHATBOT_MAX_CONTEXT_TOKENS"] = "512"
    from api.routers import chatbot as chatbot_router

    chatbot_router._reset_chatbot_modules_for_tests()
    _cleanup_test_output()

    from main import app

    transport1 = httpx.ASGITransport(app=app)
    transport2 = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport1, base_url="http://test") as c1, httpx.AsyncClient(
        transport=transport2, base_url="http://test"
    ) as c2:
        b1 = await c1.get("/chatbot/bootstrap")
        b2 = await c2.get("/chatbot/bootstrap")
        u1 = uuid.UUID(b1.json()["browser_user_id"])
        u2 = uuid.UUID(b2.json()["browser_user_id"])
        _ensure_user_in_db(u1)
        _ensure_user_in_db(u2)
        admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
        _ensure_user_in_db(admin_id)
        await _aingest_doc_for_owner(admin_id, "u1.txt", "alpha alpha alpha\n")
        await _aingest_doc_for_owner(admin_id, "u2.txt", "beta beta beta\n")

        huge = "x" * 2000
        memory = {
            "version": 0,
            "summary": "",
            "recent_messages": [{"role": "user", "content": huge} for _ in range(200)],
        }

        async def slowish_chat():
            return await c1.post(
                "/chatbot/chat",
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "message": {"role": "user", "content": "alpha"},
                    "memory": memory,
                    "options": {"include_evidence": True, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.5},
                },
            )

        async def normal_chat():
            return await c2.post(
                "/chatbot/chat",
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "message": {"role": "user", "content": "beta"},
                    "memory": {"version": 0, "summary": "", "recent_messages": []},
                    "options": {"include_evidence": True, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.5},
                },
            )

        started = time.monotonic()
        r_slow, r_fast = await asyncio.gather(slowish_chat(), normal_chat())
        elapsed = time.monotonic() - started

        assert r_slow.status_code == 413
        assert r_fast.status_code == 200
        assert elapsed < 0.7, f"large history should not serialize requests, took {elapsed:.2f}s"
