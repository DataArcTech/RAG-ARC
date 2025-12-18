import os
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import socket

    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _s.close()
except PermissionError:
    pytest.skip("Socket operations are not permitted in this environment.", allow_module_level=True)


def _ensure_env_for_chatbot_tests() -> None:
    os.environ["RAG_INFERENCE_CONFIG_PATH"] = "config/json_configs/chatbot_test/rag_inference.json"
    os.environ["KNOWLEDGE_CONFIG_PATH"] = "config/json_configs/chatbot_test/knowledge.json"
    os.environ["CHATBOT_RAG_INFERENCE_CONFIG_PATH"] = "config/json_configs/chatbot_test/rag_inference.json"
    os.environ["CHATBOT_KNOWLEDGE_CONFIG_PATH"] = "config/json_configs/chatbot_test/knowledge.json"
    os.environ["CHATBOT_MAX_CONCURRENCY"] = "16"
    os.environ["CHATBOT_MAX_CONTEXT_TOKENS"] = "64"
    os.environ["LOCAL_FILE_STORAGE_PATH"] = "./test_output/chatbot_local_files"
    os.environ["ENABLE_DEEPSEARCH"] = "0"
    os.environ["CHATBOT_CONTEXT_TURNS"] = "5"
    os.environ["CHATBOT_MAX_CONTEXT_FRACTION"] = "0.9"


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


@pytest.mark.anyio
async def test_chatbot_context_too_long_returns_sse_error():
    _ensure_env_for_chatbot_tests()
    _cleanup_test_output()

    from api.routers import chatbot as chatbot_router

    chatbot_router._reset_chatbot_modules_for_tests()

    from main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        boot = await client.get("/chatbot/bootstrap")
        assert boot.status_code == 200

        # Only required for admin ingestion; cookie user doesn't need to exist for stateless chat.
        admin_id = uuid.UUID(os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001"))
        _ensure_user_in_db(admin_id)

        conversation_id = str(uuid.uuid4())
        huge = "x" * 2000

        async with client.stream(
            "POST",
            "/api/messages",
            headers={"Accept": "text/event-stream"},
            json={
                "id": conversation_id,
                "content": "hi",
                "messages": [{"role": "user", "content": huge} for _ in range(10)],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            events = await _collect_sse_events(resp)

        assert events
        assert events[0]["type"] == "error"
        assert events[0]["code"] == 413
