import asyncio
import json
import os
import socket
import sys
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import uvicorn
from fastapi import FastAPI

# Ensure repo root is importable for modules like `app_registration`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from core.deepsearch.state import DeepSearchState
from core.deepsearch.trace import emit_trace
from framework.register import Register


class _TracingStubDeepSearchService:
    async def run(self, question: str, *, metadata=None, run_id=None, stage_listener=None, **kwargs):
        state = DeepSearchState(
            config_fingerprint="test",
            run_id=run_id or uuid.uuid4().hex,
            stage_listener=stage_listener,
        )

        await emit_trace(
            "write_outline",
            "\n".join(
                [
                    "Outline:",
                    "1) Key constraints (with evidence_ids)",
                    "2) Product highlights (with evidence_ids)",
                    "3) Risks/notes (with evidence_ids)",
                ]
            ),
            meta={"question": question},
        )

        call_id = "call_test_1"
        await emit_trace(
            "tool_call",
            json.dumps(
                {
                    "tool_name": "graph.llm_chain_explorer",
                    "call_id": call_id,
                    "plan_step": "s1",
                    "extra": {"query": "保費繳付期與保證回報條件", "channel": "graph"},
                    "descriptor": {"profile": "local", "determinism": "deterministic", "description": "Roll up graph context for the question."},
                    "routing": {"can_route_remote": True, "prefer_remote": False, "has_local": True, "default_mcp_server": "local"},
                    "coverage_metrics": {"evidence_count": 2, "coverage_ratio": 0.8, "coverage_score": 0.75, "completed_steps": 1, "total_steps": 3},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            meta={"call_id": call_id, "tool_name": "graph.llm_chain_explorer"},
        )

        await asyncio.sleep(0.01)

        await emit_trace(
            "tool_response",
            json.dumps(
                {
                    "tool_name": "graph.llm_chain_explorer",
                    "call_id": call_id,
                    "ok": True,
                    "route": "local",
                    "result": {
                        "summary": "找到繳費期與保證回報相關段落（示例）。",
                        "diagnostics": {"latency_ms": 5, "unique_match_count": 1, "match_chunk_ids": ["chunk_001"]},
                        "evidences": [
                            {
                                "chunk_id": "chunk_001",
                                "source": "sample.pdf",
                                "score": 0.91,
                                "content": "繳費期：5 年；保證回報：按條款計算（示例片段）。",
                                "provenance": {
                                    "path": "<path>/<redacted>.pdf",
                                    "page": 3,
                                    "patterns": ["繳費期", "保證回報"],
                                    "triples": [{"head": "繳費期", "relation": "為", "tail": "5年"}],
                                },
                            }
                        ],
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            meta={"call_id": call_id, "tool_name": "graph.llm_chain_explorer", "ok": True},
        )

        state.record_plan({"plan": {"plan_id": "p1", "steps": [{"step_id": "s1", "description": "rollup", "channel": "graph"}]}})
        await asyncio.sleep(0.01)
        state.record_reasoning({"reasoning_steps": [{"step_id": "s1", "status": "done"}], "evidences": []})
        await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)
        state.record_report(
            {
                "question": question,
                "answer": "stub report body (weaver tag write should show this).",
                "evidences": [],
                "highlights": [],
            }
        )
        return {
            "plan": {"plan": {"question": question, "steps": []}},
            "reasoning": {"reasoning_steps": [], "evidences": [], "coverage_metrics": {}},
            "report": {"question": question, "answer": "stub report body (weaver tag write should show this).", "evidences": [], "highlights": []},
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


def _parse_weaver_blocks(lines: list[str]) -> list[tuple[str, str]]:
    events: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line:
            if buffer:
                events.append("\n".join(buffer))
                buffer = []
            continue
        if not line.startswith("data:"):
            continue
        payload = line.split(":", 1)[1].lstrip()
        if payload == "[DONE]":
            break
        buffer.append(payload)
    blocks: list[tuple[str, str]] = []
    for event in events:
        raw_lines = [ln.rstrip("\n") for ln in event.splitlines() if ln is not None]
        if not raw_lines:
            continue
        first = raw_lines[0].strip()
        if not (first.startswith("<") and first.endswith(">")):
            continue
        tag = first[1:-1].strip()
        if not tag:
            continue
        closing = f"</{tag}>"
        if raw_lines[-1].strip() != closing:
            continue
        content = "\n".join(raw_lines[1:-1]).rstrip("\n")
        blocks.append((tag, content))
    return blocks


def test_deepsearch_weaver_trace_outputs_by_tag():
    class _StubAccount:
        def get_user_by_username(self, username: str):
            return None

    registrator = Register()
    registrator.registrations["account"] = _StubAccount()
    registrator.registrations["deepsearch_service"] = _TracingStubDeepSearchService()
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
            async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
                resp = await client.post("/deepsearch/run_async", json={"question": "保費繳付期是多少？保證回報條件是什麼？"})
                assert resp.status_code == 202
                run_id = resp.json()["run_id"]

                raw_lines: list[str] = []
                deadline = time.time() + 5.0
                async with client.stream("GET", f"/deepsearch/stream/{run_id}", params={"format": "weaver"}) as stream:
                    assert stream.status_code == 200
                    async for line in stream.aiter_lines():
                        raw_lines.append(line)
                        if time.time() > deadline:
                            raise AssertionError("Timed out waiting for weaver stream")
                        if line.startswith("data:") and line.split(":", 1)[1].strip() == "[DONE]":
                            break

                blocks = _parse_weaver_blocks(raw_lines)
                by_tag: dict[str, list[str]] = defaultdict(list)
                for tag, content in blocks:
                    by_tag[tag].append(content)

                ordered = ["think", "write_outline", "tool_call", "tool_response", "write", "progress", "terminate"]
                print("\n=== DeepSearch weaver output (grouped by tag) ===")
                for tag in ordered:
                    items = by_tag.get(tag, [])
                    print(f"\n[{tag}] count={len(items)}")
                    for idx, content in enumerate(items, start=1):
                        print(f"--- item {idx} ---")
                        if content:
                            print(content)
                        else:
                            print("(empty)")

                for tag in ordered:
                    assert by_tag.get(tag), f"missing weaver tag output: {tag}"

    asyncio.run(_run())
