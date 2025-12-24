import argparse
import os
import time
import uuid

from celery import Celery
from celery.contrib.testing.worker import start_worker

from config.encapsulation.database.cache_db.redis_config import RedisConfig


def _redis_url_from_env() -> str:
    config = RedisConfig()
    password = config.password or ""
    auth = f":{password}@" if password else ""
    db = int(getattr(config, "db", "0") or 0)
    return f"redis://{auth}{config.host}:{int(config.port)}/{db}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Celery+Redis broker stress (small defaults).")
    parser.add_argument("--n", type=int, default=60, help="Number of tasks to enqueue.")
    parser.add_argument("--sleep", type=float, default=0.01, help="Per-task sleep seconds.")
    parser.add_argument("--queue", type=str, default="", help="Queue name (defaults to a unique queue).")
    args = parser.parse_args()

    if args.n <= 0 or args.sleep < 0:
        raise SystemExit("invalid args")

    queue_name = (args.queue or "").strip() or f"mq_stress:{uuid.uuid4().hex}"

    broker_url = os.getenv("CELERY_BROKER_URL") or _redis_url_from_env()
    backend_url = os.getenv("CELERY_RESULT_BACKEND") or broker_url

    app = Celery("rag-arc-mq-stress", broker=broker_url, backend=backend_url)
    app.conf.update(
        timezone=os.getenv("TZ", "UTC"),
        enable_utc=True,
        task_track_started=True,
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_ignore_result=False,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_acks_on_failure_or_timeout=True,
        task_reject_on_worker_lost=True,
        broker_connection_retry_on_startup=True,
    )
    app.conf.broker_transport_options = {
        "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT_SECONDS", str(60))),
        "polling_interval": 1,
    }

    @app.task(bind=True, name="rag_arc_stress.sleep")  # noqa: ANN001
    def sleep_task(self, *, seconds: float) -> dict:  # noqa: ARG001
        time.sleep(max(0.0, float(seconds)))
        return {"ok": True, "seconds": float(seconds)}

    with start_worker(
        app,
        pool="solo",
        concurrency=1,
        loglevel="warning",
        queues=[queue_name],
        perform_ping_check=False,
    ):
        started = time.time()
        results = [sleep_task.apply_async(kwargs={"seconds": args.sleep}, queue=queue_name) for _ in range(args.n)]
        for res in results:
            res.get(timeout=max(30.0, args.n * args.sleep * 5))
        elapsed = time.time() - started
        print(f"done n={args.n} sleep={args.sleep} elapsed_s={elapsed:.3f} tps={args.n / max(1e-9, elapsed):.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
