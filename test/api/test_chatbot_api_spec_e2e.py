import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import socket

    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _s.close()
except PermissionError:
    pytest.skip("Socket operations are not permitted in this environment.", allow_module_level=True)


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
    os.environ["CHATBOT_CONTEXT_TURNS"] = "5"
    os.environ["CHATBOT_MAX_CONTEXT_FRACTION"] = "0.9"
    os.environ["CHATBOT_TOP_SOURCES"] = "5"
    os.environ["LOCAL_FILE_STORAGE_PATH"] = "./test_output/chatbot_local_files"
    os.environ["ENABLE_DEEPSEARCH"] = "0"


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


async def _collect_sse_events(resp: httpx.Response) -> list[dict]:
    events: list[dict] = []
    async for line in resp.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))
    return events


@pytest.fixture(scope="module")
def asgi_app():
    _ensure_env_for_chatbot_tests()
    _cleanup_test_output()

    from api.routers import chatbot as chatbot_router

    chatbot_router._reset_chatbot_modules_for_tests()

    from main import app

    return app


@pytest.mark.anyio
async def test_chatbot_sse_conversation_and_sources(asgi_app):
    transport = httpx.ASGITransport(app=asgi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        boot = await client.get("/chatbot/bootstrap")
        assert boot.status_code == 200

        admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
        _ensure_user_in_db(admin_id)
        await _aingest_doc_for_owner(
            admin_id,
            filename="facts_v2.txt",
            content="Paris is the capital of France.\n",
        )

        conversation_id = str(uuid.uuid4())
        async with client.stream(
            "POST",
            "/api/messages",
            headers={"Accept": "text/event-stream"},
            json={
                "id": conversation_id,
                "content": "What is the capital of France?",
                "messages": [],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            events = await _collect_sse_events(resp)

        chunks = [e["content"] for e in events if e.get("type") == "chunk"]
        assert chunks
        full = "".join(chunks)
        assert "ECHO:" in full

        sources = [e for e in events if e.get("type") == "sources"]
        assert sources
        assert sources[0]["id"] == conversation_id
        assert isinstance(sources[0]["sources"], list)
        assert sources[0]["sources"]
        assert sources[0]["sources"][0].get("chunk_id"), "sources[*].chunk_id required for traceability"
        assert sources[0]["sources"][0].get("file_id"), "sources[*].file_id required for traceability"
        assert sources[0]["sources"][0].get("file")
        file_url = sources[0]["sources"][0]["file"]
        assert file_url.startswith("/static/files/")
        if file_url.startswith("/"):
            file_resp = await client.get(file_url)
            assert file_resp.status_code == 200
            assert file_resp.content

        assert "<sup>1</sup>" in full

        title = [e for e in events if e.get("type") == "title"]
        assert title
        assert title[0]["id"] == conversation_id
        assert title[0]["title"]

        done = [e for e in events if e.get("type") == "done"]
        assert done
        assert done[-1]["status"] == "success"


@pytest.mark.anyio
async def test_chatbot_history_window_ignores_old_messages(asgi_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHATBOT_MAX_CONTEXT_TOKENS", "256")
    monkeypatch.setenv("CHATBOT_CONTEXT_TURNS", "5")

    transport = httpx.ASGITransport(app=asgi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        boot = await client.get("/chatbot/bootstrap")
        assert boot.status_code == 200

        admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
        _ensure_user_in_db(admin_id)
        await _aingest_doc_for_owner(admin_id, filename="window_v2.txt", content="alpha beta gamma\n")

        conversation_id = str(uuid.uuid4())
        huge = "x" * 2000

        messages = [{"role": "user", "content": huge} for _ in range(100)]
        messages.extend(
            [{"role": "assistant", "content": "ok"} if i % 2 else {"role": "user", "content": "small"} for i in range(10)]
        )

        async with client.stream(
            "POST",
            "/api/messages",
            headers={"Accept": "text/event-stream"},
            json={"id": conversation_id, "content": "alpha", "messages": messages, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            events = await _collect_sse_events(resp)

        done = [e for e in events if e.get("type") == "done"]
        assert done
        assert done[-1]["status"] == "success"


@pytest.mark.anyio
async def test_chatbot_owner_id_header_sets_cookie(asgi_app):
    transport = httpx.ASGITransport(app=asgi_app)
    owner = uuid.uuid4()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/api/messages",
            headers={"Accept": "text/event-stream", "X-Owner-Id": str(owner)},
            json={"id": str(uuid.uuid4()), "content": "hello", "messages": [], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert f"{owner}" in resp.headers.get("set-cookie", "")
            _ = await _collect_sse_events(resp)


@pytest.mark.anyio
async def test_chatbot_multi_user_concurrency(asgi_app):
    admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
    _ensure_user_in_db(admin_id)
    await _aingest_doc_for_owner(admin_id, "concurrency_v2.txt", "User content about concurrency.\n")

    transport1 = httpx.ASGITransport(app=asgi_app)
    transport2 = httpx.ASGITransport(app=asgi_app)

    async with httpx.AsyncClient(transport=transport1, base_url="http://test") as c1, httpx.AsyncClient(
        transport=transport2, base_url="http://test"
    ) as c2:
        await c1.get("/chatbot/bootstrap")
        await c2.get("/chatbot/bootstrap")

        async def ask(client: httpx.AsyncClient, text: str) -> list[dict]:
            async with client.stream(
                "POST",
                "/api/messages",
                headers={"Accept": "text/event-stream"},
                json={"id": str(uuid.uuid4()), "content": text, "messages": [], "stream": True},
            ) as resp:
                assert resp.status_code == 200
                return await _collect_sse_events(resp)

        started = time.monotonic()
        e1, e2 = await asyncio.gather(ask(c1, "Hello from user1"), ask(c2, "Hello from user2"))
        elapsed = time.monotonic() - started

        assert elapsed < 0.8, f"expected concurrency, took {elapsed:.2f}s"
        assert [e for e in e1 if e.get("type") == "done"][-1]["status"] == "success"
        assert [e for e in e2 if e.get("type") == "done"][-1]["status"] == "success"


@pytest.mark.anyio
async def test_chatbot_context_too_long_does_not_block_other_requests(asgi_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHATBOT_MAX_CONTEXT_TOKENS", "256")

    admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
    _ensure_user_in_db(admin_id)
    await _aingest_doc_for_owner(admin_id, "alpha_v2.txt", "alpha alpha alpha\n")
    await _aingest_doc_for_owner(admin_id, "beta_v2.txt", "beta beta beta\n")

    transport1 = httpx.ASGITransport(app=asgi_app)
    transport2 = httpx.ASGITransport(app=asgi_app)
    async with httpx.AsyncClient(transport=transport1, base_url="http://test") as c1, httpx.AsyncClient(
        transport=transport2, base_url="http://test"
    ) as c2:
        await c1.get("/chatbot/bootstrap")
        await c2.get("/chatbot/bootstrap")

        huge = "x" * 2000
        too_long_messages = [{"role": "user", "content": huge} for _ in range(10)]

        async def slowish_chat():
            async with c1.stream(
                "POST",
                "/api/messages",
                headers={"Accept": "text/event-stream"},
                json={"id": str(uuid.uuid4()), "content": "alpha", "messages": too_long_messages, "stream": True},
            ) as resp:
                return await _collect_sse_events(resp)

        async def normal_chat():
            async with c2.stream(
                "POST",
                "/api/messages",
                headers={"Accept": "text/event-stream"},
                json={"id": str(uuid.uuid4()), "content": "beta", "messages": [], "stream": True},
            ) as resp:
                return await _collect_sse_events(resp)

        started = time.monotonic()
        e_slow, e_fast = await asyncio.gather(slowish_chat(), normal_chat())
        elapsed = time.monotonic() - started

        assert elapsed < 0.8, f"large history should not serialize requests, took {elapsed:.2f}s"
        assert [e for e in e_slow if e.get("type") == "error"]
        assert [e for e in e_fast if e.get("type") == "done"][-1]["status"] == "success"


@pytest.mark.anyio
async def test_chatbot_static_files_are_scoped_to_shared_owner(asgi_app):
    admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
    other_owner = uuid.uuid4()
    _ensure_user_in_db(admin_id)
    _ensure_user_in_db(other_owner)

    transport = httpx.ASGITransport(app=asgi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/chatbot/bootstrap")
        file_id = await _aingest_doc_for_owner(other_owner, filename="private.txt", content="private content\n")
        resp = await client.get(f"/static/files/{file_id}/private.txt")
        assert resp.status_code == 404
