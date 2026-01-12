import os
import uuid
from typing import Any, Dict

from fastapi import HTTPException, status

from api.sse import sse_json
from core.utils.owner_guard import is_admin_owner
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue


def get_task_queue() -> RedisTaskQueue:
    return RedisTaskQueue.from_env()


def use_celery() -> bool:
    return os.getenv("TASK_QUEUE_MODE", "inprocess").strip().lower() == "celery"


def assert_task_owner(task_run: Dict[str, Any], *, user_id: uuid.UUID) -> None:
    if is_admin_owner(user_id):
        return
    raw = task_run.get("owner_id")
    if not raw:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")
    try:
        owner_uuid = uuid.UUID(str(raw))
    except Exception:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id") from None
    if owner_uuid != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")


def format_sse(*, event: str, data: Dict[str, Any], event_id: int | None = None) -> str:
    payload: Dict[str, Any] = {"event": event, "data": data}
    if event_id is not None:
        payload["id"] = event_id
    return sse_json(payload)


def normalize_file_id(file_id: str) -> str:
    """Normalize file_id by extracting the last valid UUID if it appears to be duplicated."""

    if len(file_id) <= 36:
        return file_id
    if len(file_id) >= 36:
        candidate = file_id[-36:]
        parts = candidate.split("-")
        if len(parts) == 5 and [len(p) for p in parts] == [8, 4, 4, 4, 12]:
            return candidate
    if len(file_id) >= 65:
        candidate = file_id[29 : 29 + 36]
        parts = candidate.split("-")
        if len(parts) == 5 and [len(p) for p in parts] == [8, 4, 4, 4, 12]:
            return candidate
    return file_id

