import uuid
from types import SimpleNamespace

from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, RedisTaskQueueSettings
from config.encapsulation.database.cache_db.redis_config import RedisConfig


class _FakeRedis:
    def __init__(self) -> None:
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
        if maxlen is not None and maxlen > 0:
            self._streams[stream] = self._streams[stream][-maxlen:]
        return entry_id


def test_seq_map_is_bounded_by_stream_maxlen():
    settings = RedisTaskQueueSettings(namespace=f"test:mq:{uuid.uuid4().hex}", stream_maxlen=50)
    queue = RedisTaskQueue(RedisConfig(), settings)
    fake = _FakeRedis()
    queue._redis_db = SimpleNamespace(client=fake)  # type: ignore[attr-defined]

    run_id = uuid.uuid4().hex
    for i in range(500):
        queue.append_progress_event(
            flow="t",
            task_run_id=run_id,
            stage="s",
            status="progress",
            percent=i % 100,
            resource_id=run_id,
            payload={"i": i},
        )

    seq_map_key = settings.key_seq_map(run_id)
    assert fake.zcard(seq_map_key) <= settings.stream_maxlen

    run_stream = settings.stream_progress_for_run(run_id)
    assert len(fake._streams.get(run_stream, [])) <= settings.stream_maxlen

