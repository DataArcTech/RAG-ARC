import asyncio
import os
import sys
import uuid
from types import SimpleNamespace

import pytest

from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class _FakeRedis:
    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._counters: dict[str, int] = {}
        self._tick = 0

    def ping(self):
        return True

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

    def xread(self, streams: dict[str, str], count: int = 1, block: int | None = None):  # noqa: ARG002
        (stream, last_id), *_ = list(streams.items())
        last_left, last_right = last_id.split("-", 1)
        last = (int(last_left), int(last_right))

        def _parse(s: str) -> tuple[int, int]:
            a, b = s.split("-", 1)
            return int(a), int(b)

        items = [(sid, f) for sid, f in self._streams.get(stream, []) if _parse(sid) > last]
        if not items:
            return []
        return [(stream, items[:count])]

    def xrevrange(self, stream: str, max: str = "+", min: str = "-", count: int = 1):  # noqa: A002, ARG002
        items = self._streams.get(stream, [])
        if not items:
            return []
        return list(reversed(items))[:count]


@pytest.mark.asyncio
async def test_celery_index_file_emits_stage_progress(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_QUEUE_MODE", "celery")
    monkeypatch.setenv("MQ_NAMESPACE", f"test:mq:{uuid.uuid4().hex}")

    import encapsulation.message_queue.redis_task_queue as redis_task_queue_module

    fake_client = _FakeRedis()
    fake_db = SimpleNamespace(client=fake_client)
    monkeypatch.setattr(redis_task_queue_module, "RedisDB", lambda *args, **kwargs: fake_db, raising=True)  # noqa: ARG005

    import application.knowledge.celery_tasks as knowledge_celery_tasks

    monkeypatch.setattr(knowledge_celery_tasks, "ensure_initialized", lambda: None, raising=True)
    monkeypatch.setattr(knowledge_celery_tasks, "_acquire_lock", lambda *a, **k: True, raising=True)  # noqa: ANN001
    monkeypatch.setattr(knowledge_celery_tasks, "_release_lock", lambda *a, **k: None, raising=True)  # noqa: ANN001

    owner = uuid.uuid4()

    class _Meta:
        status = None
        owner_id = owner

    class _FileStorage:
        def get_file_metadata(self, file_id: str):  # noqa: ARG002
            return _Meta()

    class _FileIndex:
        def delete_file_data(self, file_id: str, **kwargs):  # noqa: ARG002
            return {"success": True}

        async def index_file(self, file_id: str, **kwargs):  # noqa: ARG002
            progress = kwargs.get("progress")
            assert callable(progress)
            progress("retrieved", 5, {"file_id": file_id})
            await asyncio.sleep(0)
            progress("parsed", 25, {"file_id": file_id})
            await asyncio.sleep(0)
            progress("done", 100, {"file_id": file_id})
            return {"success": True, "file_id": file_id}

    class _Knowledge:
        file_storage = _FileStorage()
        file_index = _FileIndex()

    monkeypatch.setattr(knowledge_celery_tasks, "_get_knowledge", lambda: _Knowledge(), raising=True)

    run_id = uuid.uuid4().hex
    async_result = knowledge_celery_tasks.index_file.apply(kwargs={"file_id": "file-1", "owner_id": str(owner)}, task_id=run_id)
    result = async_result.get(timeout=5)
    assert result.get("success") is True

    queue = RedisTaskQueue.from_env()
    task_run = queue.get_task_run(run_id) or {}
    assert task_run.get("state") == TaskState.SUCCESS.value

    events = queue.read_progress_events(run_id, last_seq=-1, count=200, block_ms=0)
    stages = {ev.get("stage") for ev in events}
    assert {"retrieved", "parsed", "done"} <= stages
