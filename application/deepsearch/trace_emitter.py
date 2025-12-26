"""DeepSearch trace emitter helpers.

This module lives in the application layer so it can be used by both:
- FastAPI in-process execution (publish via api.deepsearch.tasks.TASKS)
- Celery execution (publish via encapsulation.message_queue.RedisTaskQueue)

Core modules emit TraceEvent objects via contextvars; this adapter converts them
into progress events stored in the existing progress event stream.
"""
import asyncio
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from core.deepsearch.trace import TraceEmitter, TraceEvent
from core.utils.json_safe import json_safe


class CallbackTraceEmitter(TraceEmitter):
    """TraceEmitter implementation backed by an async publish callback."""

    def __init__(
        self,
        *,
        run_id: str,
        publish: Callable[[str, Dict[str, Any]], Awaitable[None]],
        stage: str = "trace",
    ) -> None:
        self._run_id = str(run_id)
        self._publish = publish
        self._stage = (stage or "trace").strip() or "trace"

    async def emit(self, event: TraceEvent) -> None:
        meta = event.meta
        if is_dataclass(meta):
            meta = asdict(meta)
        payload: Dict[str, Any] = {
            "stage": self._stage,
            "trace_tag": event.tag,
            "content": str(event.content),
            "meta": json_safe(meta) if isinstance(meta, dict) else {"meta": json_safe(meta)},
        }
        await self._publish("trace", payload)


def make_inprocess_trace_emitter(
    *,
    run_id: str,
    publish: Callable[[str, Dict[str, Any]], Awaitable[None]],
) -> CallbackTraceEmitter:
    """Factory for in-process emitters that publish into the task registry/queue."""

    return CallbackTraceEmitter(run_id=run_id, publish=publish, stage="trace")

class RedisTaskQueueTraceEmitter(TraceEmitter):
    """TraceEmitter implementation that writes to RedisTaskQueue progress streams."""

    def __init__(
        self,
        *,
        run_id: str,
        task_queue: Any,
        flow: str = "deepsearch",
        resource_id: Optional[str] = None,
    ) -> None:
        self._run_id = str(run_id)
        self._task_queue = task_queue
        self._flow = (flow or "deepsearch").strip() or "deepsearch"
        self._resource_id = str(resource_id) if resource_id is not None else None

    async def emit(self, event: TraceEvent) -> None:
        meta = event.meta
        if is_dataclass(meta):
            meta = asdict(meta)
        payload: Dict[str, Any] = {
            "stage": "trace",
            "trace_tag": event.tag,
            "content": str(event.content),
            "meta": json_safe(meta) if isinstance(meta, dict) else {"meta": json_safe(meta)},
        }
        await asyncio.to_thread(
            getattr(self._task_queue, "append_progress_event"),
            flow=self._flow,
            task_run_id=self._run_id,
            stage="trace",
            status="trace",
            percent=None,
            resource_id=self._resource_id or self._run_id,
            payload=payload,
        )


def make_redis_trace_emitter(
    *,
    run_id: str,
    task_queue: Any,
    resource_id: Optional[str] = None,
) -> RedisTaskQueueTraceEmitter:
    return RedisTaskQueueTraceEmitter(
        run_id=run_id,
        task_queue=task_queue,
        resource_id=resource_id,
    )


__all__ = [
    "CallbackTraceEmitter",
    "RedisTaskQueueTraceEmitter",
    "make_inprocess_trace_emitter",
    "make_redis_trace_emitter",
]
