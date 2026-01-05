import uuid
from types import SimpleNamespace

import httpx
from fastapi import FastAPI


class _FakeRedis:
    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
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

    def hset(self, name: str, key: str, value: str):
        self._hashes.setdefault(name, {})[key] = value
        return 1

    def hget(self, name: str, key: str):
        return self._hashes.get(name, {}).get(key)

    def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True):  # noqa: ARG002
        self._tick += 1
        entry_id = f"{self._tick}-0"
        self._streams.setdefault(stream, []).append((entry_id, dict(fields)))
        if maxlen is not None and maxlen > 0:
            self._streams[stream] = self._streams[stream][-maxlen:]
        return entry_id

    def xread(self, streams: dict[str, str], count: int = 1, block: int | None = None):  # noqa: ARG002
        return []

    def xrevrange(self, stream: str, max: str = "+", min: str = "-", count: int = 1):  # noqa: A002, ARG002
        items = self._streams.get(stream, [])
        if not items:
            return []
        return list(reversed(items))[:count]


class _FakeRedisDB:
    def __init__(self, client: _FakeRedis) -> None:
        self.client = client


async def test_deepsearch_celery_endpoints_enforce_owner(monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_MODE", "celery")
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "cache+memory://")
    monkeypatch.setenv("MQ_NAMESPACE", f"test:mq:{uuid.uuid4().hex}")

    import encapsulation.message_queue.redis_task_queue as redis_task_queue_module

    fake_client = _FakeRedis()
    fake_db = _FakeRedisDB(fake_client)

    def _fake_redis_db_factory(*args, **kwargs):  # noqa: ANN001, ARG002
        return fake_db

    monkeypatch.setattr(redis_task_queue_module, "RedisDB", _fake_redis_db_factory, raising=True)

    from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState

    queue = RedisTaskQueue.from_env()

    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    run_id = uuid.uuid4().hex
    queue.create_task_run(task_run_id=run_id, task_type="deepsearch", owner_id=owner_id, resource_id=run_id)
    queue.update_task_run(
        run_id,
        state=TaskState.SUCCESS,
        progress_percent=100,
        finished=True,
        result_ref=queue.settings.key_task_result(run_id),
    )
    queue.set_task_result(run_id, {"state": {"run_id": run_id, "stage": "reported"}})

    from api.routers import deepsearch as deepsearch_router
    from api.routers.auth import get_current_user

    current = {"id": owner_id}
    app = FastAPI()
    app.include_router(deepsearch_router.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=current["id"])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        current["id"] = other_id
        assert (await client.get(f"/deepsearch/progress/{run_id}")).status_code == 403
        assert (await client.get(f"/deepsearch/result/{run_id}")).status_code == 403
        assert (await client.get(f"/deepsearch/stream/{run_id}")).status_code == 403

        current["id"] = owner_id
        resp = await client.get(f"/deepsearch/result/{run_id}")
        assert resp.status_code == 200
        assert resp.json().get("state", {}).get("run_id") == run_id
