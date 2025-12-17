from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _set_env_for_chatbot_validation() -> None:
    os.environ["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "WARNING")
    os.environ["RAG_INFERENCE_CONFIG_PATH"] = "config/json_configs/chatbot_test/rag_inference.json"
    os.environ["KNOWLEDGE_CONFIG_PATH"] = "config/json_configs/chatbot_test/knowledge.json"
    os.environ["CHATBOT_RAG_INFERENCE_CONFIG_PATH"] = "config/json_configs/chatbot_test/rag_inference.json"
    os.environ["CHATBOT_KNOWLEDGE_CONFIG_PATH"] = "config/json_configs/chatbot_test/knowledge.json"
    os.environ["LOCAL_FILE_STORAGE_PATH"] = "./test_output/chatbot_local_files"
    os.environ.setdefault("CHATBOT_MAX_CONCURRENCY", "16")
    os.environ.setdefault("CHATBOT_MAX_CONTEXT_TOKENS", "512")
    os.environ.setdefault("CHATBOT_CONTEXT_TURNS", "5")

    # Make deepsearch registration fail fast (not part of this MVP validation).
    os.environ.setdefault("DEEPSEARCH_WEB_PROVIDER", "")
    os.environ.setdefault("DEEPSEARCH_DEFAULT_ADAPTER", "")
    os.environ.setdefault("ENABLE_DEEPSEARCH", "0")


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
        if existing is not None:
            return
        session.add(
            User(
                id=user_id,
                user_name=f"chatbot_script_{str(user_id)[:8]}",
                hashed_password="chatbot-script-placeholder",
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
    result = asyncio.run(knowledge.file_index.index_file(file_id))
    if not result.get("success"):
        raise RuntimeError(f"indexing failed: {result}")
    return file_id


async def _async_concurrency_checks(app) -> None:
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

        from api.routers import chatbot as chatbot_router

        knowledge = chatbot_router._get_chatbot_knowledge()
        await asyncio.to_thread(
            knowledge.file_storage.upload_file,
            "u1.txt",
            b"alpha alpha alpha\n",
            u1,
            "text/plain",
        )
        await asyncio.to_thread(
            knowledge.file_storage.upload_file,
            "u2.txt",
            b"beta beta beta\n",
            u2,
            "text/plain",
        )

        started = time.monotonic()
        r1, r2 = await asyncio.gather(
            c1.post(
                "/chatbot/chat",
                headers={"X-Owner-Id": str(u1)},
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "message": {"role": "user", "content": "alpha"},
                    "memory": {"version": 0, "summary": "", "recent_messages": []},
                    "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
                },
            ),
            c2.post(
                "/chatbot/chat",
                headers={"X-Owner-Id": str(u2)},
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "message": {"role": "user", "content": "beta"},
                    "memory": {"version": 0, "summary": "", "recent_messages": []},
                    "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
                },
            ),
        )
        elapsed = time.monotonic() - started
        if r1.status_code != 200 or r2.status_code != 200:
            raise RuntimeError(f"concurrency chat failed: {r1.status_code} {r2.status_code}")
        if elapsed > 0.7:
            raise RuntimeError(f"expected concurrency, took {elapsed:.2f}s")

        os.environ["CHATBOT_MAX_CONTEXT_TOKENS"] = "64"
        huge = "x" * 2000
        big_memory = {"version": 0, "summary": "", "recent_messages": [{"role": "user", "content": huge} for _ in range(200)]}
        started = time.monotonic()
        r_slow, r_fast = await asyncio.gather(
            c1.post(
                "/chatbot/chat",
                headers={"X-Owner-Id": str(u1)},
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "message": {"role": "user", "content": "alpha"},
                    "memory": big_memory,
                    "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
                },
            ),
            c2.post(
                "/chatbot/chat",
                headers={"X-Owner-Id": str(u2)},
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "message": {"role": "user", "content": "beta"},
                    "memory": {"version": 0, "summary": "", "recent_messages": []},
                    "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
                },
            ),
        )
        elapsed = time.monotonic() - started
        if r_slow.status_code not in (200, 413) or r_fast.status_code != 200:
            raise RuntimeError(f"large-history test failed: {r_slow.status_code} {r_fast.status_code}")
        if elapsed > 0.7:
            raise RuntimeError(f"large history should not serialize requests, took {elapsed:.2f}s")


def main() -> int:
    _set_env_for_chatbot_validation()
    _cleanup_test_output()

    from api.routers import chatbot as chatbot_router

    chatbot_router._reset_chatbot_modules_for_tests()

    from main import app

    client_a = TestClient(app)
    client_b = TestClient(app)
    client_c = TestClient(app)
    client_anon = TestClient(app)

    a_boot = client_a.get("/chatbot/bootstrap")
    b_boot = client_b.get("/chatbot/bootstrap")
    c_boot = client_c.get("/chatbot/bootstrap")
    assert a_boot.status_code == 200
    assert b_boot.status_code == 200
    assert c_boot.status_code == 200

    owner_a = uuid.UUID(a_boot.json()["browser_user_id"])
    owner_b = uuid.UUID(b_boot.json()["browser_user_id"])
    owner_c = uuid.UUID(c_boot.json()["browser_user_id"])
    assert owner_a != owner_b

    admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
    _ensure_user_in_db(admin_id)

    file_a = _ingest_doc_for_owner(admin_id, "doc_a.txt", "alphaA only.\nParis is the capital of France.\n")
    file_b = _ingest_doc_for_owner(admin_id, "doc_b.txt", "betaB only.\nBerlin is the capital of Germany.\n")
    file_c = _ingest_doc_for_owner(admin_id, "doc_c.txt", "alphaA and betaB together.\n")
    assert file_a and file_b and file_c

    conversation_id = str(uuid.uuid4())
    mem = {"version": 0, "summary": "", "recent_messages": []}
    r1 = client_a.post(
        "/chatbot/chat",
        headers={"X-Owner-Id": str(owner_a)},
        json={
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": "alphaA"},
            "memory": mem,
            "options": {"include_evidence": True, "top_k": 3, "return_subgraph": False, "max_context_fraction": 0.5},
        },
    )
    assert r1.status_code == 200, r1.text
    data1 = r1.json()
    assert data1["citations"], "expected citations"
    assert "Sources: [1]" in data1["assistant"]["content"]
    assert data1["citations"][0]["file_id"] in {file_a, file_b, file_c}
    cite = data1["citations"][0]
    assert client_a.get(cite["chunk_url"], headers={"X-Owner-Id": str(owner_a)}).status_code == 200
    assert client_a.get(cite["file_url"], headers={"X-Owner-Id": str(owner_a)}).status_code == 200
    assert client_b.get(cite["chunk_url"], headers={"X-Owner-Id": str(owner_b)}).status_code == 200
    assert client_b.get(cite["file_url"], headers={"X-Owner-Id": str(owner_b)}).status_code == 200
    assert client_anon.get(cite["chunk_url"]).status_code == 401
    assert client_anon.get(cite["file_url"]).status_code == 401

    mem = data1["memory"]
    r2 = client_a.post(
        "/chatbot/chat",
        headers={"X-Owner-Id": str(owner_a)},
        json={
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": "Say it again."},
            "memory": mem,
            "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
        },
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["memory"]["version"] > mem["version"]

    title_resp = client_a.post(
        "/chatbot/title",
        headers={"X-Owner-Id": str(owner_a)},
        json={
            "conversation_id": conversation_id,
            "user": data1["memory"]["recent_messages"][-2]["content"] if data1["memory"]["recent_messages"] else "alphaA",
            "assistant": data1["assistant"]["content"],
        },
    )
    assert title_resp.status_code == 200, title_resp.text
    assert title_resp.json()["title"]

    c_conv = str(uuid.uuid4())
    r3 = client_c.post(
        "/chatbot/chat",
        headers={"X-Owner-Id": str(owner_c)},
        json={
            "conversation_id": c_conv,
            "message": {"role": "user", "content": "betaB"},
            "memory": {"version": 0, "summary": "", "recent_messages": []},
            "options": {"include_evidence": True, "top_k": 3, "return_subgraph": False, "max_context_fraction": 0.9},
        },
    )
    assert r3.status_code == 200, r3.text
    assert r3.json()["citations"][0]["file_id"] in {file_a, file_b, file_c}
    assert "Sources: [1]" in r3.json()["assistant"]["content"]

    ws_conv = str(uuid.uuid4())
    with client_a.websocket_connect(f"/chatbot/ws?conversation_id={ws_conv}&owner_id={owner_a}") as ws:
        ws.send_json(
            {
                "message": {"role": "user", "content": "Stream this"},
                "memory": {"version": 0, "summary": "", "recent_messages": []},
                "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
            }
        )
        start = ws.receive_json()
        assert start["type"] == "start"
        request_id = start["request_id"]
        saw_delta = False
        final = None
        for _ in range(200):
            frame = ws.receive_json()
            if frame["type"] == "delta":
                assert frame["request_id"] == request_id
                saw_delta = True
                continue
            if frame["type"] == "final":
                final = frame
                break
        assert saw_delta
        assert final is not None
        assert final["assistant"]["content"]

    # Long conversation should be supported under a single conversation_id with a fixed context window.
    os.environ["CHATBOT_MAX_CONTEXT_TOKENS"] = "512"
    os.environ["CHATBOT_CONTEXT_TURNS"] = "5"
    long_conv = str(uuid.uuid4())
    memory = {"version": 0, "summary": "", "recent_messages": []}
    for i in range(1, 16):
        r = client_a.post(
            "/chatbot/chat",
            headers={"X-Owner-Id": str(owner_a)},
            json={
                "conversation_id": long_conv,
                "message": {"role": "user", "content": f"{i}:" + ("m" * 80)},
                "memory": memory,
                "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        memory = data["memory"]
        assert len(memory.get("recent_messages") or []) <= 10

    asyncio.run(_async_concurrency_checks(app))
    print("chatbot_sandbox_validate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
