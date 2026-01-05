import uuid
from types import SimpleNamespace

from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, RedisTaskQueueSettings, TaskState
from config.encapsulation.database.cache_db.redis_config import RedisConfig


def _parse_stream_id(value: str) -> tuple[int, int]:
    left, right = value.split("-", 1)
    return int(left), int(right)


class _FakeRedis:
    def __init__(self):
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._counters: dict[str, int] = {}
        self._tick = 0

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):  # noqa: ARG002
        if nx and key in self._strings:
            return False
        self._strings[key] = value
        return True

    def get(self, key: str):
        return self._strings.get(key)

    def expire(self, key: str, ttl: int):  # noqa: ARG002
        return True

    def incr(self, key: str):
        self._counters[key] = int(self._counters.get(key, 0)) + 1
        return self._counters[key]

    def hset(self, name: str, key: str, value: str):
        self._hashes.setdefault(name, {})[key] = value
        return 1

    def hget(self, name: str, key: str):
        return self._hashes.get(name, {}).get(key)

    def zadd(self, name: str, mapping: dict[str, float | int]):
        z = self._zsets.setdefault(name, {})
        for member, score in mapping.items():
            z[str(member)] = float(score)
        return len(mapping)

    def zrangebyscore(self, name: str, min: float | int, max: float | int):  # noqa: A002
        z = self._zsets.get(name, {})
        lo = float(min)
        hi = float(max)
        items = [(m, s) for m, s in z.items() if lo <= float(s) <= hi]
        items.sort(key=lambda t: (t[1], t[0]))
        return [m for m, _ in items]

    def zcard(self, name: str):
        return len(self._zsets.get(name, {}))

    def zremrangebyrank(self, name: str, start: int, stop: int):
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
        self._tick += 1
        entry_id = f"{self._tick}-0"
        self._streams.setdefault(stream, []).append((entry_id, dict(fields)))
        return entry_id

    def xread(self, streams: dict[str, str], count: int = 1, block: int | None = None):  # noqa: ARG002
        (stream, last_id), *_ = list(streams.items())
        last = _parse_stream_id(last_id)
        items = [(sid, f) for sid, f in self._streams.get(stream, []) if _parse_stream_id(sid) > last]
        if not items:
            return []
        return [(stream, items[:count])]

    def xrevrange(self, stream: str, max: str = "+", min: str = "-", count: int = 1):  # noqa: A002, ARG002
        items = self._streams.get(stream, [])
        if not items:
            return []
        return list(reversed(items))[:count]


def test_progress_replay_uses_seq_cursor():
    settings = RedisTaskQueueSettings(namespace="test:mq")
    queue = RedisTaskQueue(RedisConfig(), settings)
    fake = _FakeRedis()
    queue._redis_db = SimpleNamespace(client=fake)  # type: ignore[attr-defined]

    owner = uuid.uuid4()
    run_id = queue.create_task_run(task_type="deepsearch", owner_id=owner, resource_id="r1")
    assert run_id

    queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=1)
    queue.append_progress_event(flow="deepsearch", task_run_id=run_id, stage="created", status="progress", percent=1, resource_id=run_id, payload={"n": 1})
    queue.append_progress_event(flow="deepsearch", task_run_id=run_id, stage="planned", status="progress", percent=10, resource_id=run_id, payload={"n": 2})

    events = queue.read_progress_events(run_id, last_seq=1, count=10, block_ms=0)
    assert len(events) == 1
    assert events[0]["seq"] == 2
    assert events[0]["payload"]["n"] == 2

    latest = queue.get_latest_progress_event(run_id)
    assert latest is not None
    assert latest["seq"] == 2


def test_progress_replay_filters_duplicates_when_seq_map_missing():
    settings = RedisTaskQueueSettings(namespace="test:mq")
    queue = RedisTaskQueue(RedisConfig(), settings)
    fake = _FakeRedis()
    queue._redis_db = SimpleNamespace(client=fake)  # type: ignore[attr-defined]

    owner = uuid.uuid4()
    run_id = queue.create_task_run(task_type="deepsearch", owner_id=owner, resource_id="r1")
    assert run_id

    queue.append_progress_event(flow="deepsearch", task_run_id=run_id, stage="created", status="progress", percent=1, resource_id=run_id, payload={"n": 1})
    queue.append_progress_event(flow="deepsearch", task_run_id=run_id, stage="planned", status="progress", percent=10, resource_id=run_id, payload={"n": 2})

    # Simulate seq_map missing (expired/flushed) while the stream still contains events.
    fake._hashes.pop(settings.key_seq_map(run_id), None)
    fake._zsets.pop(settings.key_seq_map(run_id), None)

    events = queue.read_progress_events(run_id, last_seq=1, count=10, block_ms=0)
    assert len(events) == 1
    assert events[0]["seq"] == 2


def test_progress_replay_advances_when_seq_map_missing_with_large_history():
    settings = RedisTaskQueueSettings(namespace="test:mq")
    queue = RedisTaskQueue(RedisConfig(), settings)
    fake = _FakeRedis()
    queue._redis_db = SimpleNamespace(client=fake)  # type: ignore[attr-defined]

    owner = uuid.uuid4()
    run_id = queue.create_task_run(task_type="deepsearch", owner_id=owner, resource_id="r1")
    assert run_id

    for i in range(1, 1001):
        queue.append_progress_event(
            flow="deepsearch",
            task_run_id=run_id,
            stage="progress",
            status="progress",
            percent=min(100, i // 10),
            resource_id=run_id,
            payload={"n": i},
        )

    fake._hashes.pop(settings.key_seq_map(run_id), None)
    fake._zsets.pop(settings.key_seq_map(run_id), None)

    events = queue.read_progress_events(run_id, last_seq=990, count=10, block_ms=0)
    seqs = [int(ev.get("seq", -1)) for ev in events]
    assert seqs == list(range(991, 1001))
