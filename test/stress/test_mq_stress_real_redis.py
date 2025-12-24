import os
import time
import uuid

import pytest
from celery import Celery
from celery.contrib.testing.worker import start_worker

from config.encapsulation.database.cache_db.redis_config import RedisConfig
from encapsulation.database.cache_db.redis_db import RedisDB
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState


@pytest.mark.skipif(
    os.getenv("RUN_RAGARC_MQ_STRESS_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="Set RUN_RAGARC_MQ_STRESS_TESTS=1 to enable real-Redis MQ stress smoke test.",
)
def test_mq_stress_smoke_real_redis():
    client = RedisDB(RedisConfig()).client
    assert client.ping() is True

    os.environ.setdefault("TASK_QUEUE_MODE", "celery")
    os.environ.setdefault("MQ_NAMESPACE", f"test:mq:{uuid.uuid4().hex}")
    os.environ.setdefault("MQ_TASK_RUN_TTL_SECONDS", "600")
    os.environ.setdefault("MQ_PROGRESS_TTL_SECONDS", "600")
    os.environ.setdefault("MQ_RESULT_TTL_SECONDS", "600")
    os.environ.setdefault("MQ_STREAM_MAXLEN", "200")

    config = RedisConfig()
    password = config.password or ""
    auth = f":{password}@" if password else ""
    db = int(getattr(config, "db", "0") or 0)
    broker_url = os.getenv("CELERY_BROKER_URL") or f"redis://{auth}{config.host}:{int(config.port)}/{db}"

    queue_name = f"mq_stress:{uuid.uuid4().hex}"
    app = Celery("rag-arc-mq-stress", broker=broker_url, backend="cache+memory://")
    app.conf.update(
        timezone=os.getenv("TZ", "UTC"),
        enable_utc=True,
        task_track_started=True,
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_ignore_result=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_acks_on_failure_or_timeout=True,
        task_reject_on_worker_lost=True,
        broker_connection_retry_on_startup=True,
    )
    app.conf.broker_transport_options = {"visibility_timeout": 60, "polling_interval": 1}

    task_queue = RedisTaskQueue.from_env()
    owner = uuid.UUID(int=0)

    @app.task(bind=True, name="rag_arc_stress.smoke")  # noqa: ANN001
    def smoke(self, *, run_id: str):  # noqa: ARG001
        if not task_queue.get_task_run(run_id):
            task_queue.create_task_run(task_run_id=run_id, task_type="mq_stress", owner_id=owner, resource_id=run_id)
        task_queue.append_progress_event(flow="mq_stress", task_run_id=run_id, stage="start", status="progress", percent=1, resource_id=run_id, payload={})
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result={"ok": True, "run_id": run_id},
            state=TaskState.SUCCESS,
            progress_percent=100,
            finished=True,
        )
        return {"ok": True}

    n = 200
    run_ids = [uuid.uuid4().hex for _ in range(n)]
    with start_worker(app, pool="solo", concurrency=1, loglevel="warning", queues=[queue_name], perform_ping_check=False):
        for run_id in run_ids:
            smoke.apply_async(kwargs={"run_id": run_id}, task_id=run_id, queue=queue_name)

        deadline = time.time() + 60
        pending = set(run_ids)
        while pending and time.time() < deadline:
            done_now = []
            for run_id in list(pending):
                record = task_queue.get_task_run(run_id) or {}
                if str(record.get("state") or "") == TaskState.SUCCESS.value:
                    done_now.append(run_id)
            for rid in done_now:
                pending.discard(rid)
            if pending:
                time.sleep(0.05)

        assert not pending, f"timed out waiting for {len(pending)} runs"

        # Spot-check seq_map bounding for a sample run.
        sample = run_ids[0]
        seq_map_key = task_queue.settings.key_seq_map(sample)
        assert client.zcard(seq_map_key) <= int(os.getenv("MQ_STREAM_MAXLEN", "200"))

