import asyncio
import json
import os
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import uvicorn
from fastapi import FastAPI

# Ensure repo root is importable for modules like `app_registration`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from core.deepsearch.state import DeepSearchState
from framework.register import Register


class _StubDeepSearchService:
    async def run(self, question: str, *, metadata=None, run_id=None, stage_listener=None, **kwargs):
        state = DeepSearchState(
            config_fingerprint="test",
            run_id=run_id or uuid.uuid4().hex,
            stage_listener=stage_listener,
        )
        state.record_plan({"plan": {"plan_id": "p1", "steps": []}})
        await asyncio.sleep(0.01)
        state.record_reasoning({"reasoning_steps": [], "evidences": []})
        await asyncio.sleep(0.01)
        state.record_report({"question": question, "answer": "stub", "evidences": [], "highlights": []})
        return {
            "plan": {"plan": {"question": question, "steps": []}},
            "reasoning": {"reasoning_steps": [], "evidences": [], "coverage_metrics": {}},
            "report": {"question": question, "answer": "stub", "evidences": [], "highlights": []},
            "state": state.snapshot(),
        }


class _StubRagInference:
    def get_graph_store(self):
        return None


@asynccontextmanager
async def _serve_app(app: FastAPI):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        if not server.started:
            raise RuntimeError("uvicorn server failed to start")
        yield host, port
    finally:
        server.should_exit = True
        await task


def test_deepsearch_run_async_exposes_progress_and_result():
    class _StubAccount:
        def get_user_by_username(self, username: str):
            return None

    registrator = Register()
    registrator.registrations["account"] = _StubAccount()
    registrator.registrations["deepsearch_service"] = _StubDeepSearchService()
    registrator.registrations["rag_inference"] = _StubRagInference()

    from api.routers import deepsearch as deepsearch_router
    from api.routers.auth import get_current_user

    app = FastAPI()
    app.include_router(deepsearch_router.router)
    user_id = uuid.uuid4()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)

    async def _run():
        async with _serve_app(app) as (host, port):
            base = f"http://{host}:{port}"
            async with httpx.AsyncClient(base_url=base, timeout=5.0) as client:
                resp = await client.post("/deepsearch/run_async", json={"question": "hello"})
                assert resp.status_code == 202
                run_id = resp.json()["run_id"]

                progress = await client.get(f"/deepsearch/progress/{run_id}")
                assert progress.status_code == 200

                deadline = time.time() + 3.0
                while time.time() < deadline:
                    result = await client.get(f"/deepsearch/result/{run_id}")
                    if result.status_code == 200:
                        payload = result.json()
                        assert payload.get("state", {}).get("run_id") == run_id
                        assert payload.get("state", {}).get("stage") in {"reported", "done", "failed"}
                        return
                    assert result.status_code in {409, 404}
                    await asyncio.sleep(0.05)
                raise AssertionError("Timed out waiting for async result")

    asyncio.run(_run())


def test_deepsearch_stream_emits_done_marker():
    class _StubAccount:
        def get_user_by_username(self, username: str):
            return None

    from core.deepsearch.trace import TRACE_FORMAT_VERSION, TRACE_SCHEMA_VERSION

    registrator = Register()
    registrator.registrations["account"] = _StubAccount()
    registrator.registrations["deepsearch_service"] = _StubDeepSearchService()
    registrator.registrations["rag_inference"] = _StubRagInference()

    from api.routers import deepsearch as deepsearch_router
    from api.routers.auth import get_current_user

    app = FastAPI()
    app.include_router(deepsearch_router.router)
    user_id = uuid.uuid4()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)

    async def _run():
        async with _serve_app(app) as (host, port):
            base = f"http://{host}:{port}"
            async with httpx.AsyncClient(base_url=base, timeout=5.0) as client:
                resp = await client.post("/deepsearch/run_async", json={"question": "hello"})
                assert resp.status_code == 202
                run_id = resp.json()["run_id"]

                events: list[dict] = []
                saw_done_marker = False
                async with client.stream("GET", f"/deepsearch/stream/{run_id}") as stream:
                    assert stream.status_code == 200
                    async for line in stream.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line.split(":", 1)[1].strip()
                        if data == "[DONE]":
                            saw_done_marker = True
                            break
                        events.append(json.loads(data))

                assert saw_done_marker is True
                done_events = [event for event in events if event.get("event") == "done"]
                assert done_events
                done_payload = done_events[0].get("data") or {}
                assert done_payload.get("schema_version") == TRACE_SCHEMA_VERSION
                assert done_payload.get("trace_format_version") == TRACE_FORMAT_VERSION

    asyncio.run(_run())


def test_deepsearch_stream_openai_format_emits_tool_calls():
    class _StubAccount:
        def get_user_by_username(self, username: str):
            return None

    registrator = Register()
    registrator.registrations["account"] = _StubAccount()
    registrator.registrations["deepsearch_service"] = _StubDeepSearchService()
    registrator.registrations["rag_inference"] = _StubRagInference()

    from api.routers import deepsearch as deepsearch_router
    from api.routers.auth import get_current_user

    app = FastAPI()
    app.include_router(deepsearch_router.router)
    user_id = uuid.uuid4()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)

    async def _run():
        async with _serve_app(app) as (host, port):
            base = f"http://{host}:{port}"
            async with httpx.AsyncClient(base_url=base, timeout=5.0) as client:
                resp = await client.post("/deepsearch/run_async", json={"question": "hello"})
                assert resp.status_code == 202
                run_id = resp.json()["run_id"]

                saw_done_marker = False
                saw_progress_tool_call = False

                async with client.stream("GET", f"/deepsearch/stream/{run_id}", params={"format": "openai"}) as stream:
                    assert stream.status_code == 200
                    async for line in stream.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line.split(":", 1)[1].strip()
                        if data == "[DONE]":
                            saw_done_marker = True
                            break
                        payload = json.loads(data)
                        assert payload.get("object") == "chat.completion.chunk"
                        choices = payload.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        tool_calls = delta.get("tool_calls") or []
                        if tool_calls:
                            fn = tool_calls[0].get("function") or {}
                            if fn.get("name") == "rag_arc_progress":
                                saw_progress_tool_call = True

                assert saw_done_marker is True
                assert saw_progress_tool_call is True

    asyncio.run(_run())

