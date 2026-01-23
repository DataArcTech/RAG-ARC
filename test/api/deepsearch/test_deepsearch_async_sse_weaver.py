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
from core.deepsearch.trace import emit_trace


class _StubDeepSearchServiceWithTrace:
    async def run(self, question: str, *, metadata=None, run_id=None, stage_listener=None, **kwargs):
        state = DeepSearchState(
            config_fingerprint="test",
            run_id=run_id or uuid.uuid4().hex,
            stage_listener=stage_listener,
        )
        state.record_plan({"plan": {"plan_id": "p1", "steps": [{"step_id": "plan_01", "description": "probe", "channel": "graph"}]}})
        await emit_trace(
            "tool_call",
            json.dumps({"call_id": "c1", "tool_name": "search", "plan_step": "plan_01", "extra": {"query": "test"}}),
        )
        await emit_trace(
            "tool_response",
            json.dumps(
                {
                    "call_id": "c1",
                    "tool_name": "search",
                    "result": {
                        "summary": "ok",
                        "evidences": [{"chunk_id": "ev1", "source": "graph", "content": "hello world", "score": 1.0}],
                        "diagnostics": {"latency_ms": 12},
                    },
                }
            ),
        )
        state.record_reasoning({"reasoning_steps": [{"step_id": "plan_01", "status": "done"}], "evidences": [{"chunk_id": "ev1"}]})
        await asyncio.sleep(0.01)
        state.record_report({"question": question, "answer": "stub", "evidences": [], "highlights": []})
        return {
            "plan": {"plan": {"question": question, "steps": []}},
            "reasoning": {"reasoning_steps": [], "evidences": [], "coverage_metrics": {}},
            "report": {"question": question, "answer": "stub", "evidences": [], "highlights": []},
            "state": state.snapshot(),
        }


class _StubAccount:
    def get_user_by_username(self, username: str):
        return None


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


def test_deepsearch_stream_weaver_renders_human_readable_blocks():
    from framework.register import Register

    registrator = Register()
    registrator.registrations["account"] = _StubAccount()
    registrator.registrations["deepsearch_service"] = _StubDeepSearchServiceWithTrace()
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

                saw_progress = False
                saw_all_tools = False
                saw_tool_call = False
                saw_tool_response = False
                saw_done = False
                current_tag = None
                buffer: list[str] = []
                async with client.stream("GET", f"/deepsearch/stream/{run_id}", params={"format": "weaver"}) as stream:
                    assert stream.status_code == 200
                    deadline = time.time() + 3.0
                    async for line in stream.aiter_lines():
                        if time.time() > deadline:
                            break
                        if not line.startswith("data:"):
                            continue
                        data = line.split(":", 1)[1].strip()
                        if data == "<progress>":
                            saw_progress = True
                            current_tag = "progress"
                            buffer = []
                        elif data == "<all_tools>":
                            saw_all_tools = True
                            current_tag = "all_tools"
                            buffer = []
                        elif data == "<tool_call>":
                            saw_tool_call = True
                            current_tag = "tool_call"
                            buffer = []
                        elif data == "<tool_response>":
                            saw_tool_response = True
                            current_tag = "tool_response"
                            buffer = []
                        elif data == "<terminate>":
                            current_tag = "terminate"
                            buffer = []
                        elif data == "</terminate>":
                            saw_done = True
                            break
                        elif data.startswith("</") and data.endswith(">"):
                            closing = data.strip("</>").strip()
                            if closing == current_tag:
                                body = "\n".join(buffer)
                                if current_tag == "tool_call":
                                    assert "tool=" in body
                                    assert "{" not in body
                                if current_tag == "tool_response":
                                    assert "evidence_count=" in body
                                    assert "{" not in body
                                if current_tag == "all_tools":
                                    assert "available_tools:" in body
                                buffer = []
                                current_tag = None
                        elif current_tag in {"tool_call", "tool_response", "all_tools"}:
                            buffer.append(data)

                assert saw_progress is True
                assert saw_all_tools is True
                assert saw_tool_call is True
                assert saw_tool_response is True
                assert saw_done is True

    asyncio.run(_run())
