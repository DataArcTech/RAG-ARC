import asyncio
import json
import os
import sys
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

def _seed_registry() -> None:
    """Ensure router imports do not require real DB-backed registrations."""

    from framework.register import Register

    reg = Register()
    reg.registrations.setdefault("account", object())
    reg.registrations.setdefault("chat_session", object())
    reg.registrations.setdefault("chat_message", object())
    reg.registrations.setdefault("rag_inference", object())


@pytest.fixture
def app():
    _seed_registry()
    import api.routers.rag_inference as rag_router

    app = FastAPI()
    app.include_router(rag_router.router)
    return app


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_rag_inference_stream_chat_sse_emits_message_event(monkeypatch, client, app):
    import api.routers.rag_inference as rag_router
    import api.routers.rag_inference_handlers as handlers
    import api.routers.rag_inference_modules.stream_chat.event.event_generator as event_generator
    import api.routers.rag_inference_modules.stream_chat.response.response_finalizer as response_finalizer

    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4())

    class FakeSessionHandler:
        def get_session(self, _session_id):
            return SimpleNamespace(id=_session_id, user_id=user.id)

        def update_session(self, *_args, **_kwargs):  # noqa: ANN001
            return None

    class FakeMessageHandler:
        def __init__(self):
            self._messages = []

        def create_message(self, message):
            if getattr(message, "id", None) is None:
                setattr(message, "id", uuid.uuid4())
            self._messages.append(message)
            return message

        def list_messages_by_session(self, _session_id):
            return list(self._messages)

        def update_message(self, *_args, **_kwargs):  # noqa: ANN001
            return None

    class FakeRAG:
        def stream_chat(self, query, owner_id, return_subgraph=False, **kwargs):  # noqa: ARG002
            def _gen():
                yield "assistant ok"

            chunks = []
            subgraph_data = None
            subgraph_info = None
            return (_gen(), chunks, subgraph_data, subgraph_info)

    session_handler = FakeSessionHandler()
    message_handler = FakeMessageHandler()
    rag = FakeRAG()
    monkeypatch.setattr(handlers, "_session_handler", session_handler)
    monkeypatch.setattr(handlers, "_message_handler", message_handler)
    monkeypatch.setattr(handlers, "_rag_inference_handler", rag)
    monkeypatch.setattr("api.routers.rag_inference_modules.stream_chat.utils.validators.validate_user_session", lambda *_: True)

    async def _stub_user():  # noqa: ANN001
        return user

    app.dependency_overrides[rag_router.get_current_user] = _stub_user
    try:
        async def _noop_ensure_finalization(**_kwargs):  # noqa: ANN001
            return None

        async def _stub_build_and_yield_final_response(  # noqa: ANN001
            *,
            chunk_id: str,
            model_name: str,
            created: int,
            request_id: str,
            **_kwargs,
        ):
            from api.routers.rag_inference_modules.stream_chat.event.event_generator import _yield_finish_event

            async for event in _yield_finish_event(chunk_id, model_name, created, request_id):
                yield event

        monkeypatch.setattr(response_finalizer, "_ensure_finalization", _noop_ensure_finalization, raising=True)
        monkeypatch.setattr(event_generator, "_build_and_yield_final_response", _stub_build_and_yield_final_response, raising=True)
        monkeypatch.setattr(
            response_finalizer,
            "_build_and_yield_final_response",
            _stub_build_and_yield_final_response,
            raising=True,
        )

        raw_lines: list[str] = []
        async with client.stream(
            "GET",
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        ) as stream:
            assert stream.status_code == 200
            async with asyncio.timeout(5.0):
                async for line in stream.aiter_lines():
                    raw_lines.append(line)
                    if line.startswith("data:") and line.split(":", 1)[1].strip() == "[DONE]":
                        break
    finally:
        app.dependency_overrides.pop(rag_router.get_current_user, None)

    text = "\n".join(raw_lines)
    assert '"object":"chat.completion.chunk"' in text
    assert "data: [DONE]" in text
    assert '"name":"rag_arc_progress"' in text

    parts: list[str] = []
    first_chunk = None
    progress_payload = None
    for idx, raw_line in enumerate(raw_lines):
        if idx > 500:
            break
        line = raw_line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        if len(data) > 200_000:
            continue
        envelope = json.loads(data)
        chunk = envelope.get("data") if isinstance(envelope, dict) else None
        if not isinstance(chunk, dict):
            continue
        if first_chunk is None:
            first_chunk = chunk
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        for tool_call in delta.get("tool_calls") or []:
            fn = (tool_call or {}).get("function") or {}
            if fn.get("name") == "rag_arc_progress" and progress_payload is None:
                progress_payload = json.loads(fn.get("arguments") or "{}")
        parts.append(delta.get("content") or "")
        if progress_payload is not None and "".join(parts) == "assistant ok":
            break

    assert first_chunk is not None
    first_delta = (first_chunk.get("choices") or [{}])[0].get("delta") or {}
    assert first_delta.get("role") == "assistant"
    assert progress_payload is not None
    assert progress_payload.get("v") == 1
    assert progress_payload.get("type") == "progress"
    assert progress_payload.get("stage") == "prepare"
    assert isinstance(progress_payload.get("request_id"), str) and progress_payload.get("request_id")
    assert isinstance(progress_payload.get("seq"), int) and progress_payload.get("seq") >= 1
    assert "".join(parts) == "assistant ok"


async def test_rag_inference_stream_chat_sse_requires_auth(monkeypatch, client, app):
    import api.routers.rag_inference as rag_router

    session_id = uuid.uuid4()

    async def _stub_none():  # noqa: ANN001
        return None

    app.dependency_overrides[rag_router.get_current_user] = _stub_none
    try:
        resp = await client.get(
            f"/rag_inference/stream_chat/{session_id}",
            params={"query": "hello"},
        )
    finally:
        app.dependency_overrides.pop(rag_router.get_current_user, None)

    assert resp.status_code == 401
