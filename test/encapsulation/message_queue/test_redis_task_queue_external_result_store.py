import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.encapsulation.database.cache_db.redis_config import RedisConfig
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, RedisTaskQueueSettings, TaskState


class _FakeRedisPipeline:
    def __init__(self, client, *, fail_execute: bool):  # noqa: ANN001
        self._client = client
        self._ops: list[tuple[str, tuple, dict]] = []
        self._fail_execute = bool(fail_execute)

    def set(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self._ops.append(("set", args, kwargs))
        return self

    def xadd(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self._ops.append(("xadd", args, kwargs))
        return self

    def execute(self):  # noqa: ANN001
        if self._fail_execute:
            raise RuntimeError("boom")
        for name, args, kwargs in self._ops:
            getattr(self._client, name)(*args, **kwargs)
        return True


class _FakeRedis:
    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._tick = 0
        self.fail_execute = False

    def ping(self):
        return True

    def pipeline(self, transaction: bool = True):  # noqa: ARG002
        return _FakeRedisPipeline(self, fail_execute=self.fail_execute)

    def set(self, key: str, value: str, ex: int | None = None, nx: bool = False):  # noqa: ARG002
        if nx and key in self._strings:
            return False
        self._strings[key] = value
        return True

    def get(self, key: str):
        return self._strings.get(key)

    def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True):  # noqa: ARG002
        self._tick += 1
        entry_id = f"{self._tick}-0"
        self._streams.setdefault(stream, []).append((entry_id, dict(fields)))
        if maxlen is not None and maxlen > 0:
            self._streams[stream] = self._streams[stream][-maxlen:]
        return entry_id


def _build_queue(tmp_path: Path) -> tuple[RedisTaskQueue, _FakeRedis, RedisTaskQueueSettings]:
    settings = RedisTaskQueueSettings(
        namespace=f"test:mq:{uuid.uuid4().hex}",
        stream_maxlen=100,
        result_max_inline_bytes=64,
        result_store_backend="local",
        result_store_local_dir=str(tmp_path),
    )
    queue = RedisTaskQueue(RedisConfig(), settings)
    fake = _FakeRedis()
    queue._redis_db = SimpleNamespace(client=fake)  # type: ignore[attr-defined]
    return queue, fake, settings


def test_finalize_run_externalizes_large_results_and_get_reads_back(tmp_path: Path):
    queue, fake, settings = _build_queue(tmp_path)
    run_id = "run_external_1"
    owner = uuid.UUID(int=0)
    queue.create_task_run(task_run_id=run_id, task_type="t", owner_id=owner, resource_id="r1")

    large_result = {"text": "x" * 5000}
    queue.set_task_result_and_finalize_run(
        run_id,
        result=large_result,
        state=TaskState.SUCCESS,
        progress_percent=100,
        finished=True,
    )

    task_run = queue.get_task_run(run_id)
    assert task_run is not None
    assert task_run.get("state") == TaskState.SUCCESS.value
    assert isinstance(task_run.get("result_ref"), str)
    assert task_run["result_ref"].startswith("local://")

    stored = fake.get(settings.key_task_result(run_id))
    assert stored is not None
    parsed = json.loads(stored)
    assert isinstance(parsed, dict)
    assert set(parsed.keys()) == {"__ragarc_result__"}
    assert parsed["__ragarc_result__"]["kind"] == "external"

    # Local file should exist and be readable.
    rel = parsed["__ragarc_result__"]["ref"].replace("local://", "", 1)
    assert (tmp_path / rel).exists()

    fetched = queue.get_task_result(run_id)
    assert fetched == large_result


def test_finalize_run_exec_failure_cleans_up_external_result_file(tmp_path: Path):
    queue, fake, settings = _build_queue(tmp_path)
    run_id = "run_external_2"
    owner = uuid.UUID(int=0)
    queue.create_task_run(task_run_id=run_id, task_type="t", owner_id=owner, resource_id="r1")

    fake.fail_execute = True
    queue.set_task_result_and_finalize_run(
        run_id,
        result={"text": "x" * 5000},
        state=TaskState.SUCCESS,
        progress_percent=100,
        finished=True,
    )

    # Task run state should remain non-terminal (same semantics as existing atomicity test).
    task_run = queue.get_task_run(run_id)
    assert task_run is not None
    assert task_run.get("state") == TaskState.PENDING.value
    assert queue.get_task_result(run_id) is None

    # No external file should remain.
    ns_dir = tmp_path / settings.namespace.replace(":", "_")
    if ns_dir.exists():
        assert list(ns_dir.glob("*.json")) == []

