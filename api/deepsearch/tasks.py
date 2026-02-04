"""Async DeepSearch task registry for SSE progress streaming."""
import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.sse import sse_json
from core.deepsearch.trace import with_trace_protocol
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue

logger = logging.getLogger(__name__)

def new_run_id() -> str:
    return uuid.uuid4().hex


def _now_ms() -> int:
    return int(time.time() * 1000)


def format_sse(*, event: str, data: Dict[str, Any], event_id: int | None = None) -> str:
    payload: Dict[str, Any] = {"event": event, "data": data}
    if event_id is not None:
        payload["id"] = event_id
    return sse_json(payload)


@dataclass
class DeepSearchTaskInfo:
    run_id: str
    owner_id: Optional[str] = None
    created_at_ms: int = field(default_factory=_now_ms)
    done: bool = False
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    last_progress: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    def append_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)


class DeepSearchTaskRegistry:
    """Process-local task store used for streaming progress updates."""

    def __init__(self, *, ttl_seconds: int = 1800) -> None:
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._items: Dict[str, DeepSearchTaskInfo] = {}
        self._lock = asyncio.Lock()

    async def create(self, run_id: Optional[str] = None, *, owner_id: Optional[str] = None) -> DeepSearchTaskInfo:
        run_id = run_id or new_run_id()
        info = DeepSearchTaskInfo(run_id=run_id, owner_id=str(owner_id) if owner_id else None)
        async with self._lock:
            self._items[run_id] = info
        return info

    async def get(self, run_id: str) -> Optional[DeepSearchTaskInfo]:
        await self._cleanup()
        async with self._lock:
            return self._items.get(run_id)

    async def mark_done(self, run_id: str, *, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        info = await self.get(run_id)
        if not info:
            return
        info.done = True
        info.result = result
        info.error = error
        async with info.cond:
            info.cond.notify_all()

    async def publish(self, run_id: str, *, event_type: str, payload: Dict[str, Any]) -> None:
        info = await self.get(run_id)
        if not info:
            return
        if isinstance(payload, dict):
            payload = with_trace_protocol(payload, run_id=run_id)
        else:
            payload = with_trace_protocol({"payload": payload}, run_id=run_id)
        event = {
            "id": len(info.events),
            "type": event_type,
            "timestamp_ms": _now_ms(),
            "payload": payload,
        }
        if event_type != "trace":
            info.last_progress = payload
        info.append_event(event)
        # Best-effort observability: also mirror events into RedisTaskQueue.
        # IMPORTANT: never block the event loop on Redis I/O (tests/inprocess mode rely on fast completion).
        try:
            progress = payload.get("progress") if isinstance(payload, dict) else None
            percent = None
            if isinstance(progress, dict) and "percent" in progress:
                try:
                    percent = int(progress.get("percent"))
                except Exception:
                    percent = None

            async def _append() -> None:
                try:
                    await asyncio.to_thread(
                        _get_redis_task_queue().append_progress_event,
                        flow="deepsearch",
                        task_run_id=run_id,
                        stage=str(payload.get("stage") or event_type),
                        status=str(event_type),
                        percent=percent,
                        resource_id=run_id,
                        payload=payload if isinstance(payload, dict) else {"payload": payload},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to append DeepSearch progress event to RedisTaskQueue: %s", exc, exc_info=True)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_append())
            except RuntimeError:
                # No running loop; skip Redis mirroring.
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to schedule DeepSearch progress mirroring to RedisTaskQueue: %s", exc, exc_info=True)
        async with info.cond:
            info.cond.notify_all()

    async def _cleanup(self) -> None:
        cutoff = _now_ms() - self._ttl_seconds * 1000
        async with self._lock:
            stale = [
                run_id
                for run_id, info in self._items.items()
                if info.done and info.created_at_ms < cutoff
            ]
            for run_id in stale:
                self._items.pop(run_id, None)


TASKS = DeepSearchTaskRegistry()

_TASK_QUEUE_FINGERPRINT: tuple[str, str, str, str, str] | None = None
_TASK_QUEUE: RedisTaskQueue | None = None


def _get_redis_task_queue() -> RedisTaskQueue:
    global _TASK_QUEUE, _TASK_QUEUE_FINGERPRINT
    fingerprint = (
        os.getenv("MQ_NAMESPACE", "rag-arc:mq"),
        os.getenv("MQ_TASK_RUN_TTL_SECONDS", str(24 * 3600)),
        os.getenv("MQ_PROGRESS_TTL_SECONDS", str(24 * 3600)),
        os.getenv("MQ_RESULT_TTL_SECONDS", str(24 * 3600)),
        os.getenv("MQ_STREAM_MAXLEN", "20000"),
    )
    if _TASK_QUEUE is None or fingerprint != _TASK_QUEUE_FINGERPRINT:
        _TASK_QUEUE = RedisTaskQueue.from_env()
        _TASK_QUEUE_FINGERPRINT = fingerprint
    return _TASK_QUEUE
