import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, Dict

from api.sse import sse_done
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState


async def stream_events_redis(
    task_queue: RedisTaskQueue,
    *,
    format_sse: Callable[..., str],
    run_id: str,
    last_event_id: int = -1,
) -> AsyncIterator[str]:
    task_run = task_queue.get_task_run(run_id)
    if not task_run:
        yield format_sse(event="error", data={"run_id": run_id, "message": "run_id not found"})
        yield sse_done()
        return

    cursor = max(-1, int(last_event_id))
    while True:
        task_run = task_queue.get_task_run(run_id) or {}
        state = str(task_run.get("state") or "")
        done = state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        block_ms = 0 if done else 15000
        events = await asyncio.to_thread(task_queue.read_progress_events, run_id, last_seq=cursor, count=200, block_ms=block_ms)
        for event in events:
            try:
                seq = int(event.get("seq", cursor + 1))
            except Exception:
                seq = cursor + 1
            cursor = seq
            payload = event.get("payload") or {}
            yield format_sse(
                event=str(event.get("status") or "message"),
                event_id=cursor,
                data={"run_id": run_id, **(payload if isinstance(payload, dict) else {"payload": payload})},
            )

        if done and not events:
            done_payload: Dict[str, Any] = {
                "run_id": run_id,
                "done": True,
                "error": task_run.get("error_message"),
                "progress": {"percent": task_run.get("progress_percent")} if task_run.get("progress_percent") is not None else None,
            }
            yield format_sse(event="done", data=done_payload, event_id=cursor + 1)
            yield sse_done()
            return
        if not events and not done:
            yield format_sse(event="heartbeat", data={"run_id": run_id})
            await asyncio.sleep(0)

