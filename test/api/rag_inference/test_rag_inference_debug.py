import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from encapsulation.data_model.schema import Chunk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


def _seed_registry() -> None:
    """Ensure router imports do not require real DB-backed registrations."""
    from framework.register import Register

    reg = Register()
    reg.registrations.setdefault("rag_inference", object())


@pytest.fixture
def app():
    _seed_registry()
    import api.routers.rag_inference as rag_router

    app = FastAPI()
    app.include_router(rag_router.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_rag_debug_endpoint_collects_debug(monkeypatch, client):
    import api.routers.rag_inference_handlers as handlers

    class FakeRAG:
        def _build_messages_and_context(
            self,
            *,
            query,
            owner_id,
            return_subgraph,
            progress_callback=None,
            history_text=None,
            enable_web_search=False,
            session_id=None,
            user_type=None,
            include_share=False,
            share_owner_id=None,
            debug_state=None,
            disable_intent_routing=False,
        ):
            if isinstance(debug_state, dict):
                debug_state["marker"] = {
                    "query": query,
                    "owner_id": str(owner_id),
                    "return_subgraph": bool(return_subgraph),
                    "history_text": history_text,
                    "enable_web_search": bool(enable_web_search),
                    "session_id": str(session_id) if session_id else None,
                    "user_type": user_type,
                    "include_share": bool(include_share),
                    "share_owner_id": str(share_owner_id) if share_owner_id else None,
                    "disable_intent_routing": bool(disable_intent_routing),
                }
            if progress_callback:
                progress_callback({"stage": "rewrite", "status": "end", "duration_ms": 1})
            messages = [{"role": "system", "content": "ok"}]
            chunks = [Chunk(id="c1", content="chunk", metadata={"filename": "f"})]
            return messages, chunks, None, None

    fake_rag = FakeRAG()
    monkeypatch.setattr(handlers, "_rag_inference_handler", fake_rag)

    owner_id = uuid.uuid4()
    resp = client.post(
        "/rag_inference/debug",
        json={
            "query": "hello",
            "owner_id": str(owner_id),
            "return_subgraph": False,
            "history_text": "user: hi",
            "enable_web_search": False,
            "include_share": False,
            "disable_intent_routing": True,
            "include_answer": False,
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["debug"]["marker"]["query"] == "hello"
    assert payload["debug"]["marker"]["owner_id"] == str(owner_id)
    assert payload["progress_events"][0]["stage"] == "rewrite"
    assert payload["messages"][0]["content"] == "ok"
    assert payload["chunks"][0]["content"] == "chunk"
