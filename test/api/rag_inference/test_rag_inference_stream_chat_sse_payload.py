import json
import uuid
from types import SimpleNamespace

import pytest

from encapsulation.data_model.schema import Chunk


def _iter_sse_data_lines(event: str) -> list[str]:
    lines: list[str] = []
    for raw in (event or "").splitlines():
        raw = raw.strip("\r")
        if raw.startswith("data:"):
            lines.append(raw.split(":", 1)[1].strip())
    return lines


@pytest.mark.asyncio
async def test_rag_inference_stream_chat_sse_emits_payload_tool_call(monkeypatch):
    import api.routers.rag_inference_modules.stream_chat.response.response_builder as response_builder
    import api.routers.rag_inference_modules.stream_chat.response.response_finalizer as response_finalizer

    session_id = uuid.uuid4()
    user_message = SimpleNamespace(id=uuid.uuid4())
    assistant_response = "hello world"

    evidence_payload = {"chunks": [{"id": "chunk-1"}], "summary": "evidence ok"}
    monkeypatch.setattr(response_builder, "build_chat_evidence", lambda *a, **k: evidence_payload, raising=True)

    async def _stub_build_sources_and_evidence(  # noqa: ANN001
        chunks,
        subgraph_data,
        subgraph_info,
        rag_inference_handler,
        assistant_response,
        is_fallback_response,
        **_kwargs,
    ):
        return assistant_response, [], {}

    async def _stub_get_or_create_final_assistant_message(**_kwargs):  # noqa: ANN001
        msg = SimpleNamespace(
            id=uuid.uuid4(),
            session_id=session_id,
            content={"role": "assistant", "content": assistant_response},
            created_at=None,
            subgraph_data=_kwargs.get("subgraph_data"),
            sources=[],
            source_file_ids=[],
        )
        return msg, True

    monkeypatch.setattr(response_finalizer, "build_sources_and_evidence", _stub_build_sources_and_evidence, raising=True)
    monkeypatch.setattr(
        response_finalizer,
        "_get_or_create_final_assistant_message",
        _stub_get_or_create_final_assistant_message,
        raising=True,
    )

    events: list[str] = []
    async for event in response_finalizer._build_and_yield_final_response(  # noqa: SLF001
        assistant_response=assistant_response,
        chunks=[Chunk(content="doc1", id="chunk-1")],
        subgraph_data={"nodes": ["n1"]},
        subgraph_info={"stats": 1},
        raw_llm_response=None,
        raw_mindmap_response=None,
        enable_deepsearch=False,
        deepsearch_result=None,
        deepsearch_sources_for_frontend=None,
        deepsearch_citation_key_map=None,
        return_subgraph=True,
        query="hello",
        session_id=session_id,
        first_turn=False,
        include_evidence=True,
        task_info=None,
        user_message=user_message,
        rag_inference_handler=SimpleNamespace(get_graph_store=lambda: None),
        chunk_id="chatcmpl-test",
        model_name="rag-arc",
        created=1,
        request_id="req-test",
        deepsearch_trace_file_path=None,
    ):
        events.append(event)

    payload_args = None
    outer_subgraph = None
    for event in events:
        for data in _iter_sse_data_lines(event):
            if data == "[DONE]":
                continue
            envelope = json.loads(data)
            chunk = envelope.get("data") if isinstance(envelope, dict) else None
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            for tool_call in delta.get("tool_calls") or []:
                fn = (tool_call or {}).get("function") or {}
                if fn.get("name") == "rag_arc_payload":
                    payload_args = json.loads(fn.get("arguments") or "{}")
                    outer_subgraph = chunk.get("subgraph")

    assert payload_args is not None
    assert payload_args["evidence"] == evidence_payload
    assert ((payload_args.get("message") or {}).get("content") or {}).get("content") == "hello world"
    assert outer_subgraph == {"nodes": ["n1"]}


@pytest.mark.asyncio
async def test_rag_inference_stream_chat_sse_renumbers_citations_and_sources(monkeypatch):
    import api.routers.rag_inference_modules.stream_chat.response.response_builder as response_builder

    # Make normalization deterministic (avoid suite-order dependence).
    monkeypatch.setattr(response_builder, "CITATION_STREAM_MODE", "final", raising=True)

    evidence_payload = {
        "chunks": [
            {"id": "chunk-1", "content": "doc1", "metadata": {"filename": "f1"}},
            {"id": "chunk-2", "content": "doc2", "metadata": {"filename": "f2"}},
            {"id": "chunk-3", "content": "doc3", "metadata": {"filename": "f3"}},
            {"id": "chunk-4", "content": "doc4", "metadata": {"filename": "f4"}},
            {"id": "chunk-5", "content": "doc5", "metadata": {"filename": "f5"}},
        ]
    }
    monkeypatch.setattr(response_builder, "build_chat_evidence", lambda *a, **k: evidence_payload, raising=True)

    chunks = [Chunk(content=f"doc{i}", id=f"chunk-{i}") for i in range(1, 6)]
    assistant_response = "answer<sup>1</sup> mid<sup>3</sup> end"

    final_text, sources_for_frontend, citation_key_map = await response_builder.build_sources_and_evidence(
        chunks,
        subgraph_data=None,
        subgraph_info=None,
        rag_inference_handler=SimpleNamespace(get_graph_store=lambda: None),
        assistant_response=assistant_response,
        is_fallback_response=False,
    )

    assert final_text == "answer<sup>1</sup> mid<sup>2</sup> end"
    assert [s.key for s in sources_for_frontend] == [1, 2]
    assert [s.chunk_id for s in sources_for_frontend] == ["chunk-1", "chunk-3"]
    assert citation_key_map == {1: 1, 3: 2}

