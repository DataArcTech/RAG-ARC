import uuid

import pytest

from api.routers.rag_inference_modules.stream_chat.task.task_registry import ChatTaskRegistry


@pytest.mark.asyncio
async def test_finalization_lock_is_per_task_and_reset_on_done():
    registry = ChatTaskRegistry(ttl_seconds=3600)
    info = await registry.create(
        session_id=uuid.uuid4(),
        user_message_id=uuid.uuid4(),
        query="q",
        request_id="r",
    )

    lock1 = await registry.get_finalization_lock(info.task_id)
    lock2 = await registry.get_finalization_lock(info.task_id)
    assert lock1 is lock2

    await registry.mark_done(info.task_id, assistant_message_id=uuid.uuid4())

    # mark_done drops the per-task lock; a subsequent call recreates it.
    lock3 = await registry.get_finalization_lock(info.task_id)
    assert lock3 is not lock1


@pytest.mark.asyncio
async def test_finalization_lock_is_reset_on_cancel():
    registry = ChatTaskRegistry(ttl_seconds=3600)
    info = await registry.create(
        session_id=uuid.uuid4(),
        user_message_id=uuid.uuid4(),
        query="q",
        request_id="r",
    )
    lock1 = await registry.get_finalization_lock(info.task_id)

    assert await registry.cancel(info.task_id, reason="test")
    lock2 = await registry.get_finalization_lock(info.task_id)
    assert lock2 is not lock1

