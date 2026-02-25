import asyncio
import json
import os
import socket
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from celery.contrib.testing.worker import start_worker
from fastapi import FastAPI


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1",
    reason="Requires networking + celery worker; set RUN_RAGARC_INTEGRATION_TESTS=1 to enable.",
)


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

    def expire(self, key: str, ttl: int):  # noqa: ARG002
        return True

    def incr(self, key: str):
        with self._lock:
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


@asynccontextmanager
async def _serve_app(app: FastAPI):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        if not server.started:
            raise RuntimeError("uvicorn server failed to start")
        yield host, port
    finally:
        server.should_exit = True
        await task


def test_knowledge_export_tasks_celery_mode_e2e(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_QUEUE_MODE", "celery")
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "cache+memory://")
    monkeypatch.setenv("CELERY_QUEUE_EXPORT", "indexing")
    monkeypatch.setenv("MQ_NAMESPACE", f"test:mq:{uuid.uuid4().hex}")

    import encapsulation.message_queue.redis_task_queue as redis_task_queue_module

    fake_client = _FakeRedis()
    fake_db = _FakeRedisDB(fake_client)
    monkeypatch.setattr(redis_task_queue_module, "RedisDB", lambda *args, **kwargs: fake_db, raising=True)  # noqa: ARG005

    from encapsulation.message_queue.celery_app import app as celery_app

    import application.knowledge.celery_tasks as knowledge_celery_tasks

    monkeypatch.setattr(knowledge_celery_tasks, "ensure_initialized", lambda: None, raising=True)
    monkeypatch.setattr(knowledge_celery_tasks, "_get_rag_inference", lambda: object(), raising=True)
    monkeypatch.setattr(knowledge_celery_tasks, "_get_knowledge", lambda: object(), raising=True)
    monkeypatch.setattr(knowledge_celery_tasks, "export_full_graph_payload", lambda **kwargs: {"ok": True, "type": "graph"}, raising=True)

    async def _fake_mindmap(**kwargs):  # noqa: ANN001, ARG001
        return {"tsv": "1\tRoot", "nodes": [{"id": "1 Root", "name": "Root", "category": "Root", "weight": 1}], "edges": []}

    monkeypatch.setattr(
        knowledge_celery_tasks,
        "export_file_mindmap_payload",
        _fake_mindmap,
        raising=True,
    )

    from api.routers import knowledge as knowledge_router
    from api.routers.auth import get_current_user

    user_id = uuid.uuid4()
    app = FastAPI()
    app.include_router(knowledge_router.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)

    with start_worker(
        celery_app,
        pool="solo",
        concurrency=1,
        loglevel="warning",
        queues=["indexing"],
        perform_ping_check=False,
    ):

        async def _run():
            async with _serve_app(app) as (host, port):
                base = f"http://{host}:{port}"
                async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
                    resp = await client.post("/knowledge/graph/export_async", json={"max_nodes": 10, "max_edges": 10})
                    assert resp.status_code == 202
                    payload = resp.json()
                    run_id = payload.get("run_id") or payload.get("data", {}).get("run_id")
                    assert run_id

                    saw_done = False
                    async with client.stream("GET", f"/knowledge/stream/{run_id}") as stream:
                        assert stream.status_code == 200
                        async for line in stream.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line.split(":", 1)[1].strip()
                            if data == "[DONE]":
                                saw_done = True
                                break
                    assert saw_done is True

                    result = await client.get(f"/knowledge/result/{run_id}")
                    assert result.status_code == 200
                    payload = result.json()
                    data = payload.get("data", payload) if isinstance(payload, dict) else payload
                    assert data.get("type") == "graph"

                    resp = await client.post("/knowledge/mindmap/export_async", json={"file_id": "file-1"})
                    assert resp.status_code == 202
                    payload = resp.json()
                    run_id = payload.get("run_id") or payload.get("data", {}).get("run_id")
                    assert run_id

                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        result = await client.get(f"/knowledge/result/{run_id}")
                        if result.status_code == 200:
                            payload = result.json()
                            data = payload.get("data", payload) if isinstance(payload, dict) else payload
                            assert data.get("tsv") == "1\tRoot"
                            return
                        if result.status_code not in {404, 409}:
                            raise AssertionError(f"Unexpected status {result.status_code}: {result.text}")
                        await asyncio.sleep(0.05)
                    raise AssertionError("Timed out waiting for mindmap export result")

        asyncio.run(_run())
