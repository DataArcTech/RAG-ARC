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

from core.deepsearch.state import DeepSearchState


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


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


class _StubDeepSearchService:
    async def run(self, question: str, *, metadata=None, run_id=None, stage_listener=None, **kwargs):  # noqa: ANN001, ARG002
        state = DeepSearchState(
            config_fingerprint="test",
            run_id=run_id or uuid.uuid4().hex,
            stage_listener=stage_listener,
        )
        state.record_plan({"plan": {"plan_id": "p1", "question": question, "steps": []}})
        await asyncio.sleep(0.01)
        state.record_reasoning({"question": question, "reasoning_steps": [], "evidences": []})
        await asyncio.sleep(0.01)
        state.record_gap_result({"should_trigger_external": False, "reason": "ok"})
        await asyncio.sleep(0.01)
        state.record_report({"question": question, "answer": "stub", "evidences": [], "highlights": []})
        return {
            "plan": {"plan": {"question": question, "steps": []}},
            "reasoning": {"question": question, "reasoning_steps": [], "evidences": [], "coverage_metrics": {}},
            "report": {"question": question, "answer": "stub", "evidences": [], "highlights": []},
            "state": state.snapshot(),
        }


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


def test_deepsearch_run_async_celery_mode_e2e(monkeypatch: pytest.MonkeyPatch):
    """
    Celery-mode e2e (in-process worker):
    - API schedules Celery task
    - Worker writes progress + result to RedisTaskQueue (fake redis here)
    - API exposes progress/result/SSE from RedisTaskQueue
    """

    # Ensure Celery mode branches are exercised.
    monkeypatch.setenv("TASK_QUEUE_MODE", "celery")
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "cache+memory://")
    monkeypatch.setenv("CELERY_QUEUE_DEEPSEARCH", "deepsearch")
    monkeypatch.setenv("MQ_NAMESPACE", f"test:mq:{uuid.uuid4().hex}")

    # Patch RedisTaskQueue to use a shared in-memory fake redis client.
    import encapsulation.message_queue.redis_task_queue as redis_task_queue_module

    fake_client = _FakeRedis()
    fake_db = _FakeRedisDB(fake_client)

    def _fake_redis_db_factory(*args, **kwargs):  # noqa: ANN001, ARG002
        return fake_db

    monkeypatch.setattr(redis_task_queue_module, "RedisDB", _fake_redis_db_factory, raising=True)

    from encapsulation.message_queue.celery_app import app as celery_app

    # Patch DeepSearch Celery task to avoid full app bootstrap and use a stub service.
    import application.deepsearch.celery_tasks as deepsearch_celery_tasks

    monkeypatch.setattr(deepsearch_celery_tasks, "ensure_initialized", lambda: None, raising=True)
    monkeypatch.setattr(deepsearch_celery_tasks, "_get_deepsearch_service", lambda: _StubDeepSearchService(), raising=True)
    monkeypatch.setattr(deepsearch_celery_tasks, "_get_graph_store", lambda: None, raising=True)

    from api.routers import deepsearch as deepsearch_router
    from api.routers.auth import get_current_user

    # The deepsearch router module may have been imported by other tests already (with a real Redis client).
    # Force it to use the fake redis-backed queue for this test.
    deepsearch_router.TASK_QUEUE = redis_task_queue_module.RedisTaskQueue.from_env()

    user_id = uuid.uuid4()
    app = FastAPI()
    app.include_router(deepsearch_router.router)
    # Must be stable across requests to satisfy owner-based authorization checks.
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)

    with start_worker(
        celery_app,
        pool="solo",
        concurrency=1,
        loglevel="warning",
        queues=["deepsearch"],
        perform_ping_check=False,
    ):

        async def _run():
            async with _serve_app(app) as (host, port):
                base = f"http://{host}:{port}"
                async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
                    resp = await client.post("/deepsearch/run_async", json={"question": "hello"})
                    assert resp.status_code == 202
                    run_id = resp.json()["run_id"]

                    progress = await client.get(f"/deepsearch/progress/{run_id}")
                    assert progress.status_code == 200

                    saw_done_marker = False
                    saw_progress_event = False

                    async with client.stream("GET", f"/deepsearch/stream/{run_id}") as stream:
                        assert stream.status_code == 200
                        async for line in stream.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line.split(":", 1)[1].strip()
                            if data == "[DONE]":
                                saw_done_marker = True
                                break
                            payload = json.loads(data)
                            if payload.get("event") == "progress":
                                saw_progress_event = True

                    assert saw_done_marker is True
                    assert saw_progress_event is True

                    deadline = time.time() + 3.0
                    while time.time() < deadline:
                        result = await client.get(f"/deepsearch/result/{run_id}")
                        if result.status_code == 200:
                            payload = result.json()
                            assert payload.get("state", {}).get("run_id") == run_id
                            assert payload.get("state", {}).get("stage") in {"reported", "failed"}
                            return
                        assert result.status_code in {409, 404}
                        await asyncio.sleep(0.05)
                    raise AssertionError("Timed out waiting for async result (celery mode)")

        asyncio.run(_run())
