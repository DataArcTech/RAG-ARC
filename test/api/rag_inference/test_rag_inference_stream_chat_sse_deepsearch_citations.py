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


async def test_rag_inference_stream_chat_deepsearch_emits_sup_citations_and_knowledge_links(monkeypatch, client, app):
    import api.routers.rag_inference as rag_router
    import api.routers.rag_inference_modules.stream_chat.response.response_finalizer as response_finalizer

    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4(), type=0)

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
        llm = None

        def get_graph_store(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("api.routers.rag_inference_handlers._session_handler", FakeSessionHandler())
    monkeypatch.setattr("api.routers.rag_inference_handlers._message_handler", FakeMessageHandler())
    monkeypatch.setattr("api.routers.rag_inference_handlers._rag_inference_handler", FakeRAG())
    monkeypatch.setattr("api.routers.rag_inference_modules.stream_chat.utils.validators.validate_user_session", lambda *_: True)

    deepsearch_result = {
        "plan": {"plan": {"question": "q", "steps": []}},
        "reasoning": {"reasoning_steps": [], "evidences": [], "think_notes": []},
        "report": {
            "question": "q",
            "answer": "Alpha[rep-0]. Beta[rep-1]。",
            "evidences": [
                {
                    "chunk_id": "rep-0",
                    "source": "hipporag",
                    "content": "Local snippet",
                    "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-0", "filename": "doc0.md"}}},
                },
                {
                    "chunk_id": "rep-1",
                    "source": "web.tavily",
                    "content": "Web Title\nWeb Body",
                    "provenance": {"url": "https://example.com/rep-1"},
                },
            ],
            "structured_report": {"citations": [{"evidence_id": "rep-0"}, {"evidence_id": "rep-1"}]},
        },
        "state": {},
    }

    async def _fake_process_deepsearch(*args, **kwargs):  # noqa: ANN001, ARG001
        async def _empty_gen():
            return
            yield

        return [deepsearch_result], [None], _empty_gen()

    # DeepSearch streaming uses the handler under stream_chat.deepsearch.
    monkeypatch.setattr(
        "api.routers.rag_inference_modules.stream_chat.deepsearch.deepsearch_handler.process_deepsearch",
        _fake_process_deepsearch,
    )
    # deepsearch_processor imports process_deepsearch by name; patch it to avoid calling real DeepSearch.
    monkeypatch.setattr(
        "api.routers.rag_inference_modules.stream_chat.deepsearch.deepsearch_processor.process_deepsearch",
        _fake_process_deepsearch,
    )
    # stream_chat.event.event_generator imports process_deepsearch by name, so patch it too.
    monkeypatch.setattr(
        "api.routers.rag_inference_modules.stream_chat.event.event_generator.process_deepsearch",
        _fake_process_deepsearch,
    )

    async def _stub_user():  # noqa: ANN001
        return user

    async def _stub_generate_and_update_title(*_args, **_kwargs):  # noqa: ANN001
        return None

    async def _stub_get_or_create_final_assistant_message(**_kwargs):  # noqa: ANN001
        assistant_response = str(_kwargs.get("assistant_response") or "")
        msg = SimpleNamespace(
            id=uuid.uuid4(),
            session_id=session_id,
            content={"role": "assistant", "content": assistant_response},
            created_at=None,
            subgraph_data=None,
            sources=[],
            source_file_ids=[],
        )
        return msg, True

    monkeypatch.setattr(response_finalizer, "generate_and_update_title", _stub_generate_and_update_title, raising=True)
    monkeypatch.setattr(
        response_finalizer,
        "_get_or_create_final_assistant_message",
        _stub_get_or_create_final_assistant_message,
        raising=True,
    )

    app.dependency_overrides[rag_router.get_current_user] = _stub_user
    try:
        raw_lines: list[str] = []
        async with client.stream(
            "POST",
            f"/rag_inference/stream_chat/{session_id}",
            json={"query": "q", "enable_deepsearch": True, "return_subgraph": False},
        ) as stream:
            assert stream.status_code == 200
            async with asyncio.timeout(8.0):
                async for line in stream.aiter_lines():
                    raw_lines.append(line)
                    if line.startswith("data:") and line.split(":", 1)[1].strip() == "[DONE]":
                        break
    finally:
        app.dependency_overrides.pop(rag_router.get_current_user, None)
    text = "\n".join(raw_lines)

    payload_args = None
    sources_event = None
    content_parts: list[str] = []
    for raw_line in raw_lines:
        line = raw_line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        envelope = json.loads(data)
        chunk = envelope.get("data") if isinstance(envelope, dict) else None
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") == "sources":
            sources_event = chunk
        choices = chunk.get("choices") or []
        if choices:
            delta = (choices[0] or {}).get("delta") or {}
            content_parts.append(delta.get("content") or "")
            for tool_call in delta.get("tool_calls") or []:
                fn = (tool_call or {}).get("function") or {}
                if fn.get("name") == "rag_arc_payload":
                    payload_args = json.loads(fn.get("arguments") or "{}")

    rendered = "".join(content_parts)
    assert "[rep-0]" not in rendered
    assert "[rep-1]" not in rendered
    assert "Alpha.<sup>1</sup>" in rendered
    assert "Beta。<sup>2</sup>" in rendered
    assert "## References" not in rendered
    assert "(/knowledge/chunk/rep-0)" not in rendered
    assert "(https://example.com/rep-1)" not in rendered

    assert payload_args is not None
    payload_text = ((payload_args.get("message") or {}).get("content") or {}).get("content")
    assert payload_text and "<sup>1</sup>" in payload_text
    assert "## References" not in payload_text

    assert sources_event is not None
    sources = sources_event.get("sources") or []
    assert [s.get("key") for s in sources] == [1, 2]
    assert [s.get("chunk_id") for s in sources] == ["rep-0", "rep-1"]
    assert sources[0].get("file") == "/static/files/file-0/doc0.md"
    assert sources[1].get("file") == "https://example.com/rep-1"
