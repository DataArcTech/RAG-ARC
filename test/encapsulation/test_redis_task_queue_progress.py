import json
import threading
import time
import uuid
from pathlib import Path

import pytest


def _parse_stream_id(value: str) -> tuple[int, int]:
    left, right = value.split("-", 1)
    return int(left), int(right)


class _FakeRedis:
    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._counters: dict[str, int] = {}
        self._tick = 0
        self._lock = threading.Lock()

    def ping(self):
        return True

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):  # noqa: ARG002
        with self._lock:
            if nx and key in self._strings:
                return False
            self._strings[key] = value
            return True

    def get(self, key: str):
        with self._lock:
            return self._strings.get(key)

    def delete(self, key: str):
        with self._lock:
            self._strings.pop(key, None)
            self._counters.pop(key, None)
        return 1

    def expire(self, key: str, ttl: int):  # noqa: ARG002
        return True

    def incr(self, key: str):
        with self._lock:
            if key in self._strings:
                raise RuntimeError("value is not an integer")
            self._counters[key] = int(self._counters.get(key, 0)) + 1
            return self._counters[key]

    def hset(self, name: str, key: str, value: str):
        with self._lock:
            self._hashes.setdefault(name, {})[key] = value
            return 1

    def hget(self, name: str, key: str):
        with self._lock:
            return self._hashes.get(name, {}).get(key)

    def zadd(self, name: str, mapping: dict[str, float | int]):
        with self._lock:
            z = self._zsets.setdefault(name, {})
            for member, score in mapping.items():
                z[str(member)] = float(score)
            return len(mapping)

    def zrangebyscore(self, name: str, min: float | int, max: float | int):  # noqa: A002
        with self._lock:
            z = self._zsets.get(name, {})
            lo = float(min)
            hi = float(max)
            items = [(m, s) for m, s in z.items() if lo <= float(s) <= hi]
            items.sort(key=lambda t: (t[1], t[0]))
            return [m for m, _ in items]

    def zcard(self, name: str):
        with self._lock:
            return len(self._zsets.get(name, {}))

    def zremrangebyrank(self, name: str, start: int, stop: int):
        with self._lock:
            z = self._zsets.get(name, {})
            items = sorted(z.items(), key=lambda t: (t[1], t[0]))
            n = len(items)
            if n == 0:
                return 0
            if start < 0:
                start = n + start
            if stop < 0:
                stop = n + stop
            start = max(0, start)
            stop = min(n - 1, stop)
            if start > stop:
                return 0
            to_delete = items[start : stop + 1]
            for member, _ in to_delete:
                z.pop(member, None)
            return len(to_delete)

    def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True):  # noqa: ARG002
        with self._lock:
            self._tick += 1
            entry_id = f"{self._tick}-0"
            self._streams.setdefault(stream, []).append((entry_id, dict(fields)))
            if maxlen is not None and maxlen > 0:
                self._streams[stream] = self._streams[stream][-maxlen:]
            return entry_id

    def xread(self, streams: dict[str, str], count: int = 1, block: int | None = None):
        (stream, last_id), *_ = list(streams.items())
        deadline = time.time() + (max(0, int(block)) / 1000.0 if block else 0.0)
        last = _parse_stream_id(last_id)
        while True:
            with self._lock:
                items = [(sid, f) for sid, f in self._streams.get(stream, []) if _parse_stream_id(sid) > last]
                if items:
                    return [(stream, items[:count])]
            if not block:
                return []
            if time.time() >= deadline:
                return []
            time.sleep(0.01)

    def xrevrange(self, stream: str, max: str = "+", min: str = "-", count: int = 1):  # noqa: A002, ARG002
        with self._lock:
            items = self._streams.get(stream, [])
            if not items:
                return []
            return list(reversed(items))[:count]


class _FakeRedisDB:
    def __init__(self, client: _FakeRedis) -> None:
        self.client = client


def test_progress_seq_key_corruption_is_healed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_QUEUE_MODE", "celery")
    monkeypatch.setenv("MQ_NAMESPACE", f"test:mq:{uuid.uuid4().hex}")

    import encapsulation.message_queue.redis_task_queue as redis_task_queue_module

    fake_client = _FakeRedis()
    fake_db = _FakeRedisDB(fake_client)
    monkeypatch.setattr(redis_task_queue_module, "RedisDB", lambda *args, **kwargs: fake_db, raising=True)  # noqa: ARG005

    from encapsulation.message_queue.redis_task_queue import RedisTaskQueue

    task_queue = RedisTaskQueue.from_env()
    run_id = uuid.uuid4().hex
    task_queue.create_task_run(task_run_id=run_id, task_type="test", owner_id=uuid.UUID(int=0), resource_id=run_id)

    # Corrupt seq_key to a non-int to force the healing branch.
    fake_client.set(task_queue.settings.key_seq(run_id), "oops")

    task_queue.append_progress_event(flow="t", task_run_id=run_id, stage="s", status="start", percent=1, payload={"a": 1})
    task_queue.append_progress_event(flow="t", task_run_id=run_id, stage="s", status="end", percent=100, payload={"b": 2})

    events = task_queue.read_progress_events(run_id, last_seq=-1, count=10, block_ms=0)
    seqs = [int(ev.get("seq", -999)) for ev in events]
    assert seqs == sorted(seqs)
    assert seqs[0] == 1

    # Validate last_seq replay semantics: last_seq=1 should only return seq>1.
    events2 = task_queue.read_progress_events(run_id, last_seq=1, count=10, block_ms=0)
    seqs2 = [int(ev.get("seq", -999)) for ev in events2]
    assert seqs2 == [2]

    # Spot-check payload was preserved.
    assert json.loads(json.dumps(events[0])).get("payload", {}).get("a") == 1


def test_progress_payload_externalization_roundtrips_full_payload(tmp_path: Path):
    """Large progress payloads should not break Redis; they are externalized and resolved on read."""

    from config.encapsulation.database.cache_db.redis_config import RedisConfig
    from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, RedisTaskQueueSettings

    fake_client = _FakeRedis()
    queue = RedisTaskQueue(
        RedisConfig(),
        RedisTaskQueueSettings(
            namespace=f"test:mq:{uuid.uuid4().hex}",
            stream_maxlen=100,
            # Force externalization path for progress events.
            result_max_inline_bytes=64,
            result_store_backend="local",
            result_store_local_dir=str(tmp_path),
        ),
    )
    queue._redis_db = _FakeRedisDB(fake_client)  # type: ignore[attr-defined]

    run_id = uuid.uuid4().hex
    queue.create_task_run(task_run_id=run_id, task_type="test", owner_id=uuid.UUID(int=0), resource_id=run_id)

    huge = "x" * 10000
    queue.append_progress_event(flow="t", task_run_id=run_id, stage="trace", status="trace", payload={"content": huge})

    events = queue.read_progress_events(run_id, last_seq=-1, count=10, block_ms=0)
    assert len(events) == 1
    payload = events[0].get("payload") or {}
    assert isinstance(payload, dict)
    assert payload.get("content") == huge
