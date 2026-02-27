import asyncio
import os
import sys
import time
import uuid
from types import SimpleNamespace

import httpx
import pytest
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


@pytest.mark.asyncio
async def test_deepsearch_debug_report_endpoint_returns_weaver_blocks():
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
    async def _stub_user():  # noqa: ANN001
        return SimpleNamespace(id=user_id)

    app.dependency_overrides[get_current_user] = _stub_user

    async with asyncio.timeout(5.0):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5.0) as client:
            resp = await client.post("/deepsearch/run_async", json={"question": "hello"})
            assert resp.status_code == 202
            run_id = resp.json()["run_id"]

            deadline = time.time() + 3.0
            ready = False
            while time.time() < deadline:
                result = await client.get(f"/deepsearch/result/{run_id}")
                if result.status_code == 200:
                    ready = True
                    break
                assert result.status_code in {409, 404}
                await asyncio.sleep(0.05)
            assert ready, "DeepSearch async run did not finish within deadline"

            debug = await client.get(f"/deepsearch/{run_id}/debug_report")
            assert debug.status_code == 200
            body = debug.text
            assert "DeepSearch Trace Report" in body
            assert f"run_id: {run_id}" in body
            # Weaver/XML blocks.
            assert "<think>" in body
            assert "<terminate>" in body
