"""Async DeepSearch task registry for SSE progress streaming."""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def new_run_id() -> str:
    return uuid.uuid4().hex


def _now_ms() -> int:
    return int(time.time() * 1000)


def format_sse(*, event: str, data: Dict[str, Any], event_id: int | None = None) -> str:
    payload = {
        "event": event,
        "data": data,
    }
    lines: List[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    # Keep data as compact JSON to avoid newlines inside 'data:'.
    import json

    lines.append(f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


@dataclass
class DeepSearchTaskInfo:
    run_id: str
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

    async def create(self, run_id: Optional[str] = None) -> DeepSearchTaskInfo:
        run_id = run_id or new_run_id()
        info = DeepSearchTaskInfo(run_id=run_id)
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
        event = {
            "id": len(info.events),
            "type": event_type,
            "timestamp_ms": _now_ms(),
            "payload": payload,
        }
        info.last_progress = payload
        info.append_event(event)
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

