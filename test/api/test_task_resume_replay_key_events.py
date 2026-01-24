import json
import uuid

import pytest


@pytest.mark.asyncio
async def test_replay_cached_events_replays_final_events_from_key_events(monkeypatch):
    from api.routers.rag_inference_modules.stream_chat.task.task_registry import ChatTaskInfo
    from api.routers.rag_inference_modules.stream_chat.task import task_resume

    class StubRegistry:
        async def get_redis_events(self, _task_id: str):
            return []

    monkeypatch.setattr(task_resume, "get_chat_task_registry", lambda: StubRegistry())

    session_id = uuid.uuid4()
    task_info = ChatTaskInfo(
        task_id="chat_task_test",
        session_id=session_id,
        user_message_id=uuid.uuid4(),
        query="q",
        request_id="rid",
        created_at_ms=0,
    )
    task_info.response_text = "partial"
    task_info.response_length = len(task_info.response_text)
    task_info.key_events = [
        {"type": "final_text", "timestamp_ms": 1, "data": {"type": "final_text", "content": "FINAL", "citation_key_map": {}, "id": str(session_id)}},
        {"type": "sources", "timestamp_ms": 2, "data": {"type": "sources", "sources": [], "citation_key_map": {}, "id": str(session_id)}},
        {"type": "payload_chunk", "timestamp_ms": 3, "data": {"object": "chat.completion.chunk", "choices": [], "id": "x", "created": 0, "model": "m"}},
    ]

    events = []
    async for sse in task_resume._replay_cached_events(task_info, request_id="rid2"):
        assert sse.startswith("data: ")
        payload = json.loads(sse[len("data: ") :].strip())
        events.append(payload["data"])

    assert [e.get("type") or e.get("object") for e in events] == [
        "final_text",
        "sources",
        "chat.completion.chunk",
    ]

