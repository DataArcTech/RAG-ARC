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
    os.environ.setdefault("CHATBOT_MAX_CONTEXT_FRACTION", "0.9")
    os.environ.setdefault("CHATBOT_TOP_SOURCES", "5")

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


async def _post_sse(
    client: httpx.AsyncClient,
    *,
    owner_id: uuid.UUID,
    conversation_id: str,
    content: str,
    messages: list[dict],
) -> list[dict]:
    async with client.stream(
        "POST",
        "/api/messages",
        headers={"Accept": "text/event-stream", "X-Owner-Id": str(owner_id)},
        json={"id": conversation_id, "content": content, "messages": messages, "stream": True},
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"sse request failed: {resp.status_code}")
        return await _collect_sse_events(resp)


def _events_to_answer_and_sources(events: list[dict]) -> tuple[str, list[dict]]:
    answer = "".join(e.get("content", "") for e in events if e.get("type") == "chunk")
    sources_events = [e for e in events if e.get("type") == "sources"]
    sources = (sources_events[0].get("sources") or []) if sources_events else []
    return answer, sources


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
        e1, e2 = await asyncio.gather(
            _post_sse(c1, owner_id=u1, conversation_id=str(uuid.uuid4()), content="alpha", messages=[]),
            _post_sse(c2, owner_id=u2, conversation_id=str(uuid.uuid4()), content="beta", messages=[]),
        )
        elapsed = time.monotonic() - started
        if not any(e.get("type") == "done" and e.get("status") == "success" for e in e1):
            raise RuntimeError("concurrency chat failed: u1")
        if not any(e.get("type") == "done" and e.get("status") == "success" for e in e2):
            raise RuntimeError("concurrency chat failed: u2")
        if elapsed > 0.7:
            raise RuntimeError(f"expected concurrency, took {elapsed:.2f}s")

        huge = "x" * 2000
        started = time.monotonic()
        too_long_messages = [{"role": "user", "content": huge} for _ in range(10)]
        e_slow, e_fast = await asyncio.gather(
            _post_sse(c1, owner_id=u1, conversation_id=str(uuid.uuid4()), content="alpha", messages=too_long_messages),
            _post_sse(c2, owner_id=u2, conversation_id=str(uuid.uuid4()), content="beta", messages=[]),
        )
        elapsed = time.monotonic() - started
        if not any(e.get("type") == "error" and e.get("code") == 413 for e in e_slow):
            raise RuntimeError("large-history test failed: expected 413 error event")
        if not any(e.get("type") == "done" and e.get("status") == "success" for e in e_fast):
            raise RuntimeError("large-history test failed: expected fast success")
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

    async def _run_sse_validation() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            conversation_id = str(uuid.uuid4())
            e1 = await _post_sse(client, owner_id=owner_a, conversation_id=conversation_id, content="alphaA", messages=[])
            answer1, sources1 = _events_to_answer_and_sources(e1)
            assert sources1, "expected sources"
            assert "<sup>1</sup>" in answer1, "expected sup markers"
            assert any(e.get("type") == "title" and e.get("title") for e in e1), "expected title event"

            messages = [{"role": "user", "content": "alphaA"}, {"role": "assistant", "content": answer1}]
            e2 = await _post_sse(
                client,
                owner_id=owner_a,
                conversation_id=conversation_id,
                content="Say it again.",
                messages=messages,
            )
            assert any(e.get("type") == "done" and e.get("status") == "success" for e in e2)

            c_conv = str(uuid.uuid4())
            e3 = await _post_sse(client, owner_id=owner_c, conversation_id=c_conv, content="betaB", messages=[])
            answer3, sources3 = _events_to_answer_and_sources(e3)
            assert sources3, "expected sources"
            assert "<sup>1</sup>" in answer3

    asyncio.run(_run_sse_validation())

    # Long conversation should be supported when the frontend keeps full history
    # while the backend only uses the last 5 turns.
    async def _run_long_conversation() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            long_conv = str(uuid.uuid4())
            history: list[dict] = []
            for i in range(1, 16):
                e = await _post_sse(
                    client,
                    owner_id=owner_a,
                    conversation_id=long_conv,
                    content=f"{i}:" + ("m" * 80),
                    messages=history,
                )
                answer, _sources = _events_to_answer_and_sources(e)
                history.append({"role": "user", "content": f"{i}:" + ("m" * 80)})
                history.append({"role": "assistant", "content": answer})
                assert any(ev.get("type") == "done" and ev.get("status") == "success" for ev in e)

    asyncio.run(_run_long_conversation())

    asyncio.run(_async_concurrency_checks(app))
    print("chatbot_sandbox_validate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
