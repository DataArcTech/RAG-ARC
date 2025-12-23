import enum
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from config.encapsulation.database.cache_db.redis_config import RedisConfig
from encapsulation.database.cache_db.redis_db import RedisDB

logger = logging.getLogger(__name__)


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


def _utc_from_ms(ts_ms: int) -> str:
    return datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + "Z"


class TaskState(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELED = "CANCELED"


@dataclass(frozen=True)
class RedisTaskQueueSettings:
    namespace: str = "rag-arc:mq"
    task_run_ttl_seconds: int = 24 * 3600
    progress_ttl_seconds: int = 24 * 3600
    result_ttl_seconds: int = 24 * 3600
    stream_maxlen: int = 20000

    task_run_stream_key: str = "task_runs"
    progress_stream_key: str = "progress_events"

    def stream_task_runs(self) -> str:
        return f"{self.namespace}:stream:{self.task_run_stream_key}"

    def stream_progress(self) -> str:
        return f"{self.namespace}:stream:{self.progress_stream_key}"

    def stream_progress_for_run(self, task_run_id: str) -> str:
        return f"{self.namespace}:stream:{self.progress_stream_key}:{task_run_id}"

    def key_task_run(self, task_run_id: str) -> str:
        return f"{self.namespace}:task_run:{task_run_id}"

    def key_resource_latest(self, task_type: str, resource_id: str) -> str:
        return f"{self.namespace}:resource_latest:{task_type}:{resource_id}"

    def key_seq(self, task_run_id: str) -> str:
        return f"{self.namespace}:seq:{task_run_id}"

    def key_seq_map(self, task_run_id: str) -> str:
        return f"{self.namespace}:seq_map:{task_run_id}"

    def key_task_result(self, task_run_id: str) -> str:
        return f"{self.namespace}:task_result:{task_run_id}"


class RedisTaskQueue:
    """
    Lightweight task run + progress event store in Redis.

    - Task runs stored as JSON at `...:task_run:<task_run_id>`
    - Task run upserts appended to Redis Stream `...:stream:task_runs`
    - Progress events appended to Redis Stream `...:stream:progress_events`
    - Progress events also appended to per-run stream `...:stream:progress_events:<task_run_id>`
    - Optional resource->latest task pointer at `...:resource_latest:<task_type>:<resource_id>`
    """

    def __init__(self, redis_config: RedisConfig, settings: RedisTaskQueueSettings):
        self._redis_config = redis_config
        self._settings = settings
        self._redis_db: RedisDB | None = None

    @staticmethod
    def _fail_fast_on_unavailable_redis() -> bool:
        explicit = os.getenv("MQ_FAILFAST_ON_REDIS_DOWN", "").strip().lower()
        if explicit in {"1", "true", "yes"}:
            return True
        if explicit in {"0", "false", "no"}:
            return False
        # Default: fail-fast in celery mode so tasks/status don't silently disappear.
        return os.getenv("TASK_QUEUE_MODE", "inprocess").strip().lower() == "celery"

    @classmethod
    def from_env(cls) -> "RedisTaskQueue":
        settings = RedisTaskQueueSettings(
            namespace=os.getenv("MQ_NAMESPACE", "rag-arc:mq"),
            task_run_ttl_seconds=int(os.getenv("MQ_TASK_RUN_TTL_SECONDS", str(24 * 3600))),
            progress_ttl_seconds=int(os.getenv("MQ_PROGRESS_TTL_SECONDS", str(24 * 3600))),
            result_ttl_seconds=int(os.getenv("MQ_RESULT_TTL_SECONDS", str(24 * 3600))),
            stream_maxlen=int(os.getenv("MQ_STREAM_MAXLEN", "20000")),
        )
        return cls(RedisConfig(), settings)

    @property
    def settings(self) -> RedisTaskQueueSettings:
        return self._settings

    def _client(self):
        if self._redis_db is not None:
            return self._redis_db.client
        try:
            self._redis_db = RedisDB(self._redis_config)
            return self._redis_db.client
        except Exception as exc:
            if self._fail_fast_on_unavailable_redis():
                raise RuntimeError(f"RedisTaskQueue unavailable: {exc}") from exc
            logger.warning("RedisTaskQueue disabled (Redis not available): %s", exc)
            self._redis_db = None
            return None

    def _xadd(self, stream: str, fields: Dict[str, str]) -> Optional[str]:
        client = self._client()
        if client is None:
            return None
        try:
            entry_id = client.xadd(stream, fields, maxlen=self._settings.stream_maxlen, approximate=True)
            return entry_id
        except Exception as exc:
            logger.warning("RedisTaskQueue xadd failed (%s): %s", stream, exc)
            return None

    def create_task_run(
        self,
        *,
        task_run_id: Optional[str] = None,
        task_type: str,
        owner_id: uuid.UUID,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_run_id = (task_run_id or "").strip() or uuid.uuid4().hex
        now_ms = _utc_now_ms()
        record: Dict[str, Any] = {
            "task_run_id": task_run_id,
            "task_type": task_type,
            "owner_id": str(owner_id),
            "resource_id": resource_id,
            "state": TaskState.PENDING.value,
            "progress_percent": 0,
            "created_at_ms": now_ms,
            "updated_at_ms": now_ms,
            "finished_at_ms": None,
            "error_message": None,
            "result_ref": None,
            "metadata": metadata or {},
        }
        self.upsert_task_run(record, resource_id=resource_id)
        return task_run_id

    def upsert_task_run(self, record: Dict[str, Any], *, resource_id: Optional[str] = None) -> None:
        client = self._client()
        if client is None:
            return
        task_run_id = str(record.get("task_run_id") or "")
        if not task_run_id:
            return
        key = self._settings.key_task_run(task_run_id)
        try:
            client.set(key, json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            client.expire(key, self._settings.task_run_ttl_seconds)
        except Exception as exc:
            logger.warning("RedisTaskQueue set task_run failed (%s): %s", task_run_id, exc)
            return

        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._xadd(self._settings.stream_task_runs(), {"task_run_id": task_run_id, "payload": payload})

        if resource_id and record.get("task_type"):
            try:
                pointer_key = self._settings.key_resource_latest(str(record["task_type"]), str(resource_id))
                client.set(pointer_key, task_run_id)
                client.expire(pointer_key, self._settings.task_run_ttl_seconds)
            except Exception as exc:
                logger.warning("RedisTaskQueue set resource pointer failed: %s", exc)

    def get_task_run(self, task_run_id: str) -> Optional[Dict[str, Any]]:
        client = self._client()
        if client is None:
            return None
        try:
            raw = client.get(self._settings.key_task_run(task_run_id))
            if not raw:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("RedisTaskQueue get task_run failed (%s): %s", task_run_id, exc)
            return None

    def get_latest_task_run_id_for_resource(self, *, task_type: str, resource_id: str) -> Optional[str]:
        client = self._client()
        if client is None:
            return None
        try:
            return client.get(self._settings.key_resource_latest(task_type, resource_id))
        except Exception as exc:
            logger.warning("RedisTaskQueue get resource pointer failed: %s", exc)
            return None

    def update_task_run(
        self,
        task_run_id: str,
        *,
        state: TaskState,
        progress_percent: Optional[int] = None,
        error_message: Optional[str] = None,
        result_ref: Optional[str] = None,
        finished: bool = False,
        metadata_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self.get_task_run(task_run_id) or {
            "task_run_id": task_run_id,
            "task_type": None,
            "owner_id": None,
            "resource_id": None,
            "state": TaskState.PENDING.value,
            "progress_percent": 0,
            "created_at_ms": _utc_now_ms(),
            "updated_at_ms": _utc_now_ms(),
            "finished_at_ms": None,
            "error_message": None,
            "result_ref": None,
            "metadata": {},
        }
        now_ms = _utc_now_ms()
        # Never downgrade a terminal state (e.g. SUCCESS -> RUNNING) due to late progress updates.
        previous_state = str(record.get("state") or "")
        is_terminal = previous_state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        if is_terminal and state.value != previous_state:
            record["state"] = previous_state
        else:
            record["state"] = state.value
        record["updated_at_ms"] = now_ms
        if progress_percent is not None:
            record["progress_percent"] = max(0, min(100, int(progress_percent)))
        if error_message is not None:
            record["error_message"] = error_message
        if result_ref is not None:
            record["result_ref"] = result_ref
        if finished:
            record["finished_at_ms"] = now_ms
        if metadata_patch:
            meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            meta.update(metadata_patch)
            record["metadata"] = meta
        self.upsert_task_run(record, resource_id=record.get("resource_id"))

    def append_progress_event(
        self,
        *,
        flow: str,
        task_run_id: str,
        stage: str,
        status: str,
        percent: Optional[int] = None,
        resource_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        client = self._client()
        if client is None:
            return None
        seq_key = self._settings.key_seq(task_run_id)
        try:
            seq = int(client.incr(seq_key))
            client.expire(seq_key, self._settings.progress_ttl_seconds)
        except Exception:
            seq = 0

        ts_ms = _utc_now_ms()
        event: Dict[str, Any] = {
            "v": 1,
            "flow": flow,
            "run_id": task_run_id,
            "resource_id": resource_id,
            "seq": seq,
            "ts_ms": ts_ms,
            "ts": _utc_from_ms(ts_ms),
            "stage": stage,
            "status": status,
            "percent": None if percent is None else max(0, min(100, int(percent))),
            "payload": payload or {},
        }
        event_payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        # Global stream for archival / Postgres sync.
        self._xadd(self._settings.stream_progress(), {"task_run_id": task_run_id, "payload": event_payload})

        # Per-run stream for efficient SSE replay.
        run_stream = self._settings.stream_progress_for_run(task_run_id)
        entry_id = self._xadd(run_stream, {"task_run_id": task_run_id, "payload": event_payload})
        # Prevent unbounded growth of per-run stream keys.
        try:
            client.expire(run_stream, self._settings.progress_ttl_seconds)
        except Exception:
            pass
        if entry_id:
            try:
                seq_map_key = self._settings.key_seq_map(task_run_id)
                client.hset(seq_map_key, str(seq), entry_id)
                client.expire(seq_map_key, self._settings.progress_ttl_seconds)
            except Exception:
                pass
        return entry_id

    def set_task_result(self, task_run_id: str, result: Dict[str, Any]) -> None:
        client = self._client()
        if client is None:
            return
        key = self._settings.key_task_result(task_run_id)
        try:
            client.set(key, json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            client.expire(key, self._settings.result_ttl_seconds)
        except Exception as exc:
            logger.warning("RedisTaskQueue set task result failed (%s): %s", task_run_id, exc)

    def get_task_result(self, task_run_id: str) -> Optional[Dict[str, Any]]:
        client = self._client()
        if client is None:
            return None
        key = self._settings.key_task_result(task_run_id)
        try:
            raw = client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("RedisTaskQueue get task result failed (%s): %s", task_run_id, exc)
            return None

    def read_progress_events(
        self,
        task_run_id: str,
        *,
        last_seq: int = -1,
        count: int = 200,
        block_ms: int = 15000,
    ) -> list[Dict[str, Any]]:
        client = self._client()
        if client is None:
            return []

        start_id = "0-0"
        if last_seq >= 0:
            try:
                mapped = client.hget(self._settings.key_seq_map(task_run_id), str(last_seq))
                if mapped:
                    start_id = mapped
            except Exception:
                start_id = "0-0"

        stream = self._settings.stream_progress_for_run(task_run_id)
        try:
            if block_ms and block_ms > 0:
                res = client.xread({stream: start_id}, count=max(1, int(count)), block=int(block_ms))
            else:
                res = client.xread({stream: start_id}, count=max(1, int(count)))
        except Exception as exc:
            logger.warning("RedisTaskQueue xread progress failed (%s): %s", task_run_id, exc)
            return []

        if not res:
            return []
        entries = res[0][1]
        events: list[Dict[str, Any]] = []
        for _, fields in entries:
            payload = fields.get("payload")
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            # When seq_map is missing (expired/flushed), start_id may fall back to "0-0".
            # Filter by seq to avoid replaying already delivered events.
            if last_seq >= 0:
                try:
                    seq_val = int(parsed.get("seq"))  # type: ignore[arg-type]
                except Exception:
                    seq_val = None
                if seq_val is not None and seq_val <= last_seq:
                    continue
            events.append(parsed)
        return events

    def get_latest_progress_event(self, task_run_id: str) -> Optional[Dict[str, Any]]:
        client = self._client()
        if client is None:
            return None
        stream = self._settings.stream_progress_for_run(task_run_id)
        try:
            entries = client.xrevrange(stream, max="+", min="-", count=1)
        except Exception as exc:
            logger.warning("RedisTaskQueue xrevrange progress failed (%s): %s", task_run_id, exc)
            return None
        if not entries:
            return None
        _, fields = entries[0]
        payload = fields.get("payload")
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
