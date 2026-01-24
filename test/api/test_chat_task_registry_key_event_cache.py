import uuid

import pytest


@pytest.mark.asyncio
async def test_append_event_always_caches_sources_and_final_events():
    from api.routers.rag_inference_modules.stream_chat.task.task_registry import ChatTaskRegistry

    registry = ChatTaskRegistry(ttl_seconds=3600, redis_client=None, use_redis_events=False)
    task = await registry.create(
        session_id=uuid.uuid4(),
        user_message_id=uuid.uuid4(),
        query="q",
        request_id="rid",
    )

    # Fill the key-event ring buffer to capacity with non-critical events.
    for _ in range(registry._MAX_KEY_EVENTS):
        await registry.append_event(task.task_id, {"delta": {"content": "x"}}, event_type="data")

    info = await registry.get(task.task_id)
    assert info is not None
    assert len(info.key_events) == registry._MAX_KEY_EVENTS

    # Even when full, critical render events must still be retained for resume.
    await registry.append_event(task.task_id, {"type": "sources", "sources": []}, event_type="sources")
    await registry.append_event(task.task_id, {"type": "final_text", "content": "hi"}, event_type="final_text")
    await registry.append_event(task.task_id, {"object": "chat.completion.chunk"}, event_type="payload_chunk")

    info = await registry.get(task.task_id)
    assert info is not None
    types = [ev.get("type") for ev in info.key_events]
    assert "sources" in types
    assert "final_text" in types
    assert "payload_chunk" in types

