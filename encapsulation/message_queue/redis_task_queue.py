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
from encapsulation.message_queue.result_store import ResultStore, ResultStoreError, build_result_store

logger = logging.getLogger(__name__)
_RESULT_ENVELOPE_KEY = "__ragarc_result__"
_PROGRESS_PAYLOAD_ENVELOPE_KEY = "__ragarc_progress_payload__"

_APPEND_PROGRESS_EVENT_LUA = r"""
-- KEYS:
--   1) seq_key
--   2) seq_map_key
--   3) global_stream
--   4) run_stream
-- ARGV:
--   1) event_json (must be valid JSON; field `seq` will be overwritten)
--   2) ttl_seconds
--   3) stream_maxlen
local seq_key = KEYS[1]
local seq_map_key = KEYS[2]
local global_stream = KEYS[3]
local run_stream = KEYS[4]
local event_json = ARGV[1]
local ttl = tonumber(ARGV[2]) or 0
local maxlen = tonumber(ARGV[3]) or 0

local ok, seq = pcall(redis.call, "INCR", seq_key)
if not ok then
  redis.call("DEL", seq_key)
  seq = redis.call("INCR", seq_key)
end

if ttl > 0 then
  redis.call("EXPIRE", seq_key, ttl)
end

local event = cjson.decode(event_json)
event["seq"] = seq
local payload = cjson.encode(event)
local run_id = tostring(event["run_id"] or "")

local args = {"*", "task_run_id", run_id, "payload", payload}
if maxlen > 0 then
  redis.call("XADD", global_stream, "MAXLEN", "~", maxlen, unpack(args))
  local entry_id = redis.call("XADD", run_stream, "MAXLEN", "~", maxlen, unpack(args))
  redis.call("ZADD", seq_map_key, seq, entry_id)
  local card = redis.call("ZCARD", seq_map_key)
  if card > maxlen then
    redis.call("ZREMRANGEBYRANK", seq_map_key, 0, card - maxlen - 1)
  end
  if ttl > 0 then
    redis.call("EXPIRE", run_stream, ttl)
    redis.call("EXPIRE", seq_map_key, ttl)
  end
  return {seq, entry_id}
end

redis.call("XADD", global_stream, unpack(args))
local entry_id = redis.call("XADD", run_stream, unpack(args))
redis.call("ZADD", seq_map_key, seq, entry_id)
if ttl > 0 then
  redis.call("EXPIRE", run_stream, ttl)
  redis.call("EXPIRE", seq_map_key, ttl)
end
return {seq, entry_id}
"""


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
    result_max_inline_bytes: int = 256 * 1024
    result_store_backend: str = "local"
    result_store_local_dir: str = "local/mq_results"
    result_store_minio_endpoint: str | None = None
    result_store_minio_bucket: str | None = None
    stream_maxlen: int = 20000
    progress_payload_max_string_chars: int = 4000
    progress_payload_max_list_items: int = 200
    progress_payload_max_depth: int = 6

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
        self._result_store: ResultStore | None = None

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
            result_max_inline_bytes=int(os.getenv("MQ_RESULT_MAX_INLINE_BYTES", str(256 * 1024))),
            result_store_backend=os.getenv("MQ_RESULT_STORE", "local"),
            result_store_local_dir=os.getenv("MQ_RESULT_LOCAL_DIR", "local/mq_results"),
            result_store_minio_endpoint=os.getenv("MQ_RESULT_MINIO_ENDPOINT") or None,
            result_store_minio_bucket=os.getenv("MQ_RESULT_MINIO_BUCKET") or None,
            stream_maxlen=int(os.getenv("MQ_STREAM_MAXLEN", "20000")),
            progress_payload_max_string_chars=int(os.getenv("MQ_PROGRESS_MAX_STRING_CHARS", "4000")),
            progress_payload_max_list_items=int(os.getenv("MQ_PROGRESS_MAX_LIST_ITEMS", "200")),
            progress_payload_max_depth=int(os.getenv("MQ_PROGRESS_MAX_DEPTH", "6")),
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

    def _get_result_store(self) -> ResultStore:
        if self._result_store is not None:
            return self._result_store
        self._result_store = build_result_store(
            backend=self._settings.result_store_backend,
            local_dir=self._settings.result_store_local_dir,
            minio_endpoint=self._settings.result_store_minio_endpoint,
            minio_bucket=self._settings.result_store_minio_bucket,
        )
        return self._result_store

    def _should_externalize_result(self, *, payload_size_bytes: int) -> bool:
        limit = int(self._settings.result_max_inline_bytes or 0)
        if limit <= 0:
            return False
        return int(payload_size_bytes) > limit

    def _result_envelope(self, *, ref: str, size_bytes: int) -> Dict[str, Any]:
        return {
            _RESULT_ENVELOPE_KEY: {
                "v": 1,
                "kind": "external",
                "ref": str(ref),
                "size_bytes": int(size_bytes),
            }
        }

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

    def _seq_map_lookup_stream_id(self, client, *, seq_map_key: str, seq: int) -> Optional[str]:  # noqa: ANN001
        # Preferred mapping: ZSET score=seq member=stream_id
        try:
            members = client.zrangebyscore(seq_map_key, seq, seq)
            if members:
                return str(members[0])
        except Exception:
            pass
        # Back-compat: HASH seq->stream_id
        try:
            mapped = client.hget(seq_map_key, str(seq))
            if mapped:
                return str(mapped)
        except Exception:
            return None
        return None

    def _seq_map_set(self, client, *, seq_map_key: str, seq: int, stream_id: str) -> None:  # noqa: ANN001
        maxlen = int(self._settings.stream_maxlen or 0)
        try:
            client.zadd(seq_map_key, {str(stream_id): int(seq)})
            if maxlen > 0:
                try:
                    card = int(client.zcard(seq_map_key))
                    if card > maxlen:
                        client.zremrangebyrank(seq_map_key, 0, card - maxlen - 1)
                except Exception:
                    pass
            return
        except Exception:
            pass
        # Back-compat / minimal clients: best-effort hash (may grow unbounded; used only when ZSET unsupported).
        try:
            client.hset(seq_map_key, str(seq), str(stream_id))
        except Exception:
            return

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
            client.set(
                key,
                json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                ex=self._settings.task_run_ttl_seconds,
            )
        except Exception as exc:
            logger.warning("RedisTaskQueue set task_run failed (%s): %s", task_run_id, exc)
            return

        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._xadd(self._settings.stream_task_runs(), {"task_run_id": task_run_id, "payload": payload})

        if resource_id and record.get("task_type"):
            try:
                pointer_key = self._settings.key_resource_latest(str(record["task_type"]), str(resource_id))
                client.set(pointer_key, task_run_id, ex=self._settings.task_run_ttl_seconds)
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

    def set_task_result_and_finalize_run(
        self,
        task_run_id: str,
        *,
        result: Dict[str, Any],
        state: TaskState,
        progress_percent: Optional[int] = None,
        error_message: Optional[str] = None,
        finished: bool = False,
        metadata_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Atomically write `task_result` and the corresponding terminal TaskRun update (MULTI/EXEC).

        This prevents partial states such as:
        - state=SUCCESS but missing result
        - result exists but state remains PENDING/RUNNING
        """
        client = self._client()
        if client is None:
            return

        result_key = self._settings.key_task_result(task_run_id)

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
        if finished:
            record["finished_at_ms"] = now_ms
        if metadata_patch:
            meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            meta.update(metadata_patch)
            record["metadata"] = meta

        task_run_key = self._settings.key_task_run(task_run_id)
        result_payload_inline = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        payload_size_bytes = len(result_payload_inline.encode("utf-8", errors="replace"))
        external_ref: str | None = None
        result_payload_to_redis = result_payload_inline
        record_result_ref = result_key
        if self._should_externalize_result(payload_size_bytes=payload_size_bytes):
            try:
                external_ref = self._get_result_store().put_bytes(
                    namespace=self._settings.namespace,
                    run_id=task_run_id,
                    payload=result_payload_inline.encode("utf-8", errors="replace"),
                    ttl_seconds=int(self._settings.result_ttl_seconds),
                )
                result_payload_to_redis = json.dumps(
                    self._result_envelope(ref=external_ref, size_bytes=payload_size_bytes),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                record_result_ref = external_ref
            except Exception as exc:  # noqa: BLE001
                logger.warning("RedisTaskQueue external result store failed; falling back to inline: %s", exc)
                external_ref = None
                result_payload_to_redis = result_payload_inline
                record_result_ref = result_key

        record["result_ref"] = record_result_ref
        task_run_payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

        try:
            try:
                pipe = client.pipeline(transaction=True)
            except Exception:
                pipe = None

            if pipe is None:
                # Best-effort fallback for minimal clients (tests/fakes) that lack pipelines.
                client.set(result_key, result_payload_to_redis, ex=self._settings.result_ttl_seconds)
                client.set(task_run_key, task_run_payload, ex=self._settings.task_run_ttl_seconds)
                try:
                    client.xadd(
                        self._settings.stream_task_runs(),
                        {"task_run_id": task_run_id, "payload": task_run_payload},
                        maxlen=self._settings.stream_maxlen,
                        approximate=True,
                    )
                except Exception:
                    pass
                resource_id = record.get("resource_id")
                task_type = record.get("task_type")
                if resource_id and task_type:
                    try:
                        client.set(
                            self._settings.key_resource_latest(str(task_type), str(resource_id)),
                            task_run_id,
                            ex=self._settings.task_run_ttl_seconds,
                        )
                    except Exception:
                        pass
                return

            pipe.set(result_key, result_payload_to_redis, ex=self._settings.result_ttl_seconds)
            pipe.set(task_run_key, task_run_payload, ex=self._settings.task_run_ttl_seconds)
            pipe.xadd(
                self._settings.stream_task_runs(),
                {"task_run_id": task_run_id, "payload": task_run_payload},
                maxlen=self._settings.stream_maxlen,
                approximate=True,
            )
            resource_id = record.get("resource_id")
            task_type = record.get("task_type")
            if resource_id and task_type:
                pipe.set(
                    self._settings.key_resource_latest(str(task_type), str(resource_id)),
                    task_run_id,
                    ex=self._settings.task_run_ttl_seconds,
                )
            pipe.execute()
        except Exception as exc:
            if external_ref:
                try:
                    self._get_result_store().delete(external_ref)
                except Exception:
                    pass
            logger.warning("RedisTaskQueue finalize_run failed (%s): %s", task_run_id, exc)
            return

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

        def _trim_value(value: Any, *, depth: int) -> tuple[Any, bool]:
            if depth <= 0:
                return ("…", True)
            if value is None:
                return (None, False)
            if isinstance(value, (bool, int, float)):
                return (value, False)
            if isinstance(value, str):
                limit = max(0, int(self._settings.progress_payload_max_string_chars))
                if limit and len(value) > limit:
                    return (value[:limit].rstrip() + "…", True)
                return (value, False)
            if isinstance(value, bytes):
                limit = max(0, int(self._settings.progress_payload_max_string_chars))
                preview = value[:limit].decode("utf-8", errors="replace") if limit else ""
                return (preview + ("…" if limit and len(value) > limit else ""), True)
            if isinstance(value, dict):
                changed_any = False
                trimmed: Dict[str, Any] = {}
                for k, v in value.items():
                    item, changed = _trim_value(v, depth=depth - 1)
                    trimmed[str(k)] = item
                    changed_any = changed_any or changed
                return (trimmed, changed_any)
            if isinstance(value, (list, tuple, set)):
                items = list(value)
                max_items = max(0, int(self._settings.progress_payload_max_list_items))
                if max_items:
                    kept = items[:max_items]
                    dropped = len(items) - len(kept)
                else:
                    kept = []
                    dropped = len(items)
                trimmed_list: list[Any] = []
                changed_any = dropped > 0
                for item_value in kept:
                    item, changed = _trim_value(item_value, depth=depth - 1)
                    trimmed_list.append(item)
                    changed_any = changed_any or changed
                return (trimmed_list, changed_any)
            return (str(value), True)

        payload_value: Any = payload or {}
        trimmed_payload, payload_trimmed = _trim_value(payload_value, depth=int(self._settings.progress_payload_max_depth))
        if payload_trimmed:
            if isinstance(trimmed_payload, dict):
                trimmed_payload.setdefault("_mq_truncated", True)
            else:
                trimmed_payload = {"payload": trimmed_payload, "_mq_truncated": True}

        ts_ms = _utc_now_ms()
        event: Dict[str, Any] = {
            "v": 1,
            "flow": flow,
            "run_id": task_run_id,
            "resource_id": resource_id,
            # seq is assigned by Redis and overwritten by the Lua fast-path.
            "seq": 0,
            "ts_ms": ts_ms,
            "ts": _utc_from_ms(ts_ms),
            "stage": stage,
            "status": status,
            "percent": None if percent is None else max(0, min(100, int(percent))),
            "payload": trimmed_payload,
        }
        event_payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
        max_inline = int(self._settings.result_max_inline_bytes or 0)
        if max_inline > 0:
            payload_size_bytes = len(event_payload.encode("utf-8", errors="replace"))
            if payload_size_bytes > max_inline:
                # Keep Redis healthy by externalizing oversize payloads, while preserving the full (untrimmed)
                # payload for later inspection. This mirrors the result externalization pattern, but applies to
                # progress/trace events as well.
                try:
                    original_payload_json = json.dumps(payload_value, ensure_ascii=False, separators=(",", ":"), default=str)
                    original_size_bytes = len(original_payload_json.encode("utf-8", errors="replace"))
                    ref = self._get_result_store().put_bytes(
                        namespace=self._settings.namespace,
                        run_id=f"{task_run_id}:progress:{uuid.uuid4().hex}",
                        payload=original_payload_json.encode("utf-8", errors="replace"),
                        ttl_seconds=int(self._settings.progress_ttl_seconds),
                    )
                    event["payload"] = {
                        _PROGRESS_PAYLOAD_ENVELOPE_KEY: {
                            "v": 1,
                            "kind": "external",
                            "ref": str(ref),
                            "size_bytes": int(original_size_bytes),
                            "note": "payload externalized to protect Redis",
                        }
                    }
                    event_payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Progress event payload too large; externalization failed; truncating further (flow=%s run_id=%s size_bytes=%d limit_bytes=%d): %s",
                        flow,
                        task_run_id,
                        payload_size_bytes,
                        max_inline,
                        exc,
                    )
                    event["payload"] = {
                        "_mq_truncated": True,
                        "_mq_original_size_bytes": payload_size_bytes,
                        "_mq_limit_bytes": max_inline,
                        "note": "payload truncated to protect Redis (externalization failed)",
                        "keys": sorted(list(payload_value.keys()))[:200] if isinstance(payload_value, dict) else None,
                    }
                    event_payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)

        # Fast-path: one atomic Redis roundtrip (seq increment + stream writes + seq_map + TTLs).
        try:
            res = client.eval(
                _APPEND_PROGRESS_EVENT_LUA,
                4,
                self._settings.key_seq(task_run_id),
                self._settings.key_seq_map(task_run_id),
                self._settings.stream_progress(),
                self._settings.stream_progress_for_run(task_run_id),
                event_payload,
                str(int(self._settings.progress_ttl_seconds)),
                str(int(self._settings.stream_maxlen)),
            )
            if isinstance(res, (list, tuple)) and len(res) == 2:
                entry_id = res[1]
                return str(entry_id) if entry_id else None
        except Exception:
            # Fallback below (best-effort).
            pass

        # Fallback path (best-effort, non-atomic).
        seq_key = self._settings.key_seq(task_run_id)
        # Defensive: seq_key can be corrupted to a non-integer value; reset on INCR errors.
        try:
            seq = int(client.incr(seq_key))
        except Exception:
            try:
                client.delete(seq_key)
                seq = int(client.incr(seq_key))
            except Exception:
                seq = 0
        try:
            client.expire(seq_key, self._settings.progress_ttl_seconds)
        except Exception:
            pass
        event["seq"] = seq
        event_payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))

        # Global stream for archival / Postgres sync.
        self._xadd(self._settings.stream_progress(), {"task_run_id": task_run_id, "payload": event_payload})

        run_stream = self._settings.stream_progress_for_run(task_run_id)
        entry_id = self._xadd(run_stream, {"task_run_id": task_run_id, "payload": event_payload})
        try:
            client.expire(run_stream, self._settings.progress_ttl_seconds)
        except Exception:
            pass
        if entry_id:
            try:
                seq_map_key = self._settings.key_seq_map(task_run_id)
                self._seq_map_set(client, seq_map_key=seq_map_key, seq=seq, stream_id=str(entry_id))
                client.expire(seq_map_key, self._settings.progress_ttl_seconds)
            except Exception:
                pass
        return entry_id

    def set_task_result(self, task_run_id: str, result: Dict[str, Any]) -> None:
        client = self._client()
        if client is None:
            return
        key = self._settings.key_task_result(task_run_id)
        result_payload_inline = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        payload_size_bytes = len(result_payload_inline.encode("utf-8", errors="replace"))
        external_ref: str | None = None
        value_to_set = result_payload_inline
        if self._should_externalize_result(payload_size_bytes=payload_size_bytes):
            try:
                external_ref = self._get_result_store().put_bytes(
                    namespace=self._settings.namespace,
                    run_id=task_run_id,
                    payload=result_payload_inline.encode("utf-8", errors="replace"),
                    ttl_seconds=int(self._settings.result_ttl_seconds),
                )
                value_to_set = json.dumps(
                    self._result_envelope(ref=external_ref, size_bytes=payload_size_bytes),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("RedisTaskQueue external result store failed; falling back to inline: %s", exc)
                external_ref = None
                value_to_set = result_payload_inline
        try:
            client.set(
                key,
                value_to_set,
                ex=self._settings.result_ttl_seconds,
            )
        except Exception as exc:
            if external_ref:
                try:
                    self._get_result_store().delete(external_ref)
                except Exception:
                    pass
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
            parsed = json.loads(raw)
            if (
                isinstance(parsed, dict)
                and set(parsed.keys()) == {_RESULT_ENVELOPE_KEY}
                and isinstance(parsed.get(_RESULT_ENVELOPE_KEY), dict)
            ):
                meta = parsed.get(_RESULT_ENVELOPE_KEY) or {}
                kind = str(meta.get("kind") or "").strip().lower()
                ref = meta.get("ref")
                if kind == "external" and isinstance(ref, str) and ref.strip():
                    try:
                        return self._get_result_store().get_json(ref)
                    except ResultStoreError as exc:
                        logger.warning("RedisTaskQueue external result read failed (%s): %s", task_run_id, exc)
                        return None
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("RedisTaskQueue external result read unexpected error (%s): %s", task_run_id, exc)
                        return None
            return parsed if isinstance(parsed, dict) else None
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

        seq_map_key = self._settings.key_seq_map(task_run_id)
        stream = self._settings.stream_progress_for_run(task_run_id)

        start_id = "0-0"
        needs_seek = False
        if last_seq >= 0:
            try:
                mapped = self._seq_map_lookup_stream_id(client, seq_map_key=seq_map_key, seq=int(last_seq))
                if mapped:
                    start_id = mapped
                else:
                    needs_seek = True
            except Exception:
                needs_seek = True

        def _parse_entries(entries: list[tuple[str, dict[str, str]]]) -> list[Dict[str, Any]]:
            parsed_events: list[Dict[str, Any]] = []
            for entry_id, fields in entries:
                payload = fields.get("payload")
                if not payload:
                    continue
                try:
                    parsed = json.loads(payload)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue

                # Resolve externally-stored progress payloads (e.g. trace tool responses with full-page evidence).
                try:
                    payload_obj = parsed.get("payload")
                    if (
                        isinstance(payload_obj, dict)
                        and set(payload_obj.keys()) == {_PROGRESS_PAYLOAD_ENVELOPE_KEY}
                        and isinstance(payload_obj.get(_PROGRESS_PAYLOAD_ENVELOPE_KEY), dict)
                    ):
                        meta = payload_obj.get(_PROGRESS_PAYLOAD_ENVELOPE_KEY) or {}
                        kind = str(meta.get("kind") or "").strip().lower()
                        ref = meta.get("ref")
                        if kind == "external" and isinstance(ref, str) and ref.strip():
                            try:
                                resolved = self._get_result_store().get_json(ref)
                                parsed["payload"] = resolved if isinstance(resolved, dict) else {"payload": resolved}
                                parsed.setdefault("_mq_external_payload_resolved", True)
                            except ResultStoreError as exc:
                                logger.warning("RedisTaskQueue external progress payload read failed (%s): %s", task_run_id, exc)
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("RedisTaskQueue external progress payload read unexpected error (%s): %s", task_run_id, exc)
                except Exception:
                    # Never break progress streaming on payload resolution failures.
                    pass

                if last_seq >= 0:
                    try:
                        seq_val = int(parsed.get("seq"))  # type: ignore[arg-type]
                    except Exception:
                        seq_val = None
                    if seq_val is not None:
                        if needs_seek:
                            try:
                                self._seq_map_set(client, seq_map_key=seq_map_key, seq=int(seq_val), stream_id=str(entry_id))
                            except Exception:
                                pass
                        if seq_val <= last_seq:
                            continue
                parsed_events.append(parsed)
            if needs_seek:
                try:
                    client.expire(seq_map_key, self._settings.progress_ttl_seconds)
                except Exception:
                    pass
            return parsed_events

        def _xread_from(start: str, *, block: int, read_count: int | None = None) -> list[tuple[str, dict[str, str]]]:
            effective_count = max(1, int(read_count if read_count is not None else count))
            try:
                if block and block > 0:
                    res = client.xread({stream: start}, count=effective_count, block=int(block))
                else:
                    res = client.xread({stream: start}, count=effective_count)
            except Exception as exc:
                logger.warning("RedisTaskQueue xread progress failed (%s): %s", task_run_id, exc)
                return []
            if not res:
                return []
            return list(res[0][1] or [])

        if not needs_seek:
            entries = _xread_from(start_id, block=int(block_ms))
            return _parse_entries(entries)

        cursor_id = start_id
        scan_count = min(2000, max(200, int(count)))
        for _ in range(200):
            entries = _xread_from(cursor_id, block=0, read_count=scan_count)
            if not entries:
                break
            cursor_id = entries[-1][0]
            events = _parse_entries(entries)
            if events:
                return events[: max(1, int(count))]

        entries = _xread_from(cursor_id, block=int(block_ms))
        return _parse_entries(entries)

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

    def get_latest_progress_event_filtered(
        self,
        task_run_id: str,
        *,
        exclude_statuses: set[str],
        scan_count: int = 200,
    ) -> Optional[Dict[str, Any]]:
        """Return the newest progress event whose status is not excluded."""

        client = self._client()
        if client is None:
            return None
        stream = self._settings.stream_progress_for_run(task_run_id)
        try:
            entries = client.xrevrange(stream, max="+", min="-", count=max(1, int(scan_count)))
        except Exception as exc:
            logger.warning("RedisTaskQueue xrevrange progress failed (%s): %s", task_run_id, exc)
            return None
        if not entries:
            return None
        excluded = {str(item) for item in exclude_statuses if str(item).strip()}
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
            status = str(parsed.get("status") or "").strip()
            if status and status in excluded:
                continue
            return parsed
        return None
