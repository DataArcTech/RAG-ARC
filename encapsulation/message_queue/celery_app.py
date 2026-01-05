import os

from celery import Celery
from dotenv import load_dotenv

from config.encapsulation.database.cache_db.redis_config import RedisConfig


def _redis_url_from_config(config: RedisConfig) -> str:
    password = config.password or ""
    auth = f":{password}@" if password else ""
    db = int(getattr(config, "db", "0") or 0)
    return f"redis://{auth}{config.host}:{int(config.port)}/{db}"


def _default_broker_url() -> str:
    config = RedisConfig()
    return _redis_url_from_config(config)


def _default_result_backend() -> str:
    return os.getenv("CELERY_RESULT_BACKEND") or _default_broker_url()


load_dotenv()

broker_url = os.getenv("CELERY_BROKER_URL") or _default_broker_url()
result_backend = _default_result_backend()

app = Celery(
    "rag-arc",
    broker=broker_url,
    backend=result_backend,
    include=[
        "application.knowledge.celery_tasks",
        "application.deepsearch.celery_tasks",
    ],
)

app.conf.update(
    timezone=os.getenv("TZ", "UTC"),
    enable_utc=True,
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Prefer explicit result storage (RedisTaskQueue) for long tasks; disable backend writes by default.
    task_ignore_result=os.getenv("CELERY_TASK_IGNORE_RESULT", "true").lower() in {"1", "true", "yes"},
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES_SECONDS", str(3600))),
    # Avoid one worker process hoarding many tasks (important for long tasks).
    worker_prefetch_multiplier=int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1")),
    task_acks_late=os.getenv("CELERY_TASK_ACKS_LATE", "true").lower() in {"1", "true", "yes"},
    task_acks_on_failure_or_timeout=os.getenv("CELERY_ACKS_ON_FAILURE_OR_TIMEOUT", "true").lower() in {"1", "true", "yes"},
    task_reject_on_worker_lost=os.getenv("CELERY_REJECT_ON_WORKER_LOST", "true").lower() in {"1", "true", "yes"},
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT_SECONDS", "0")) or None,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT_SECONDS", "0")) or None,
    broker_connection_retry_on_startup=True,
)

app.conf.broker_transport_options = {
    # Redis broker "visibility timeout" (in seconds) should exceed the longest task runtime when acks_late=True.
    "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT_SECONDS", str(24 * 3600))),
    # How long the consumer blocks between polls (seconds; Redis BRPOP requires an integer).
    "polling_interval": max(1, int(os.getenv("CELERY_REDIS_POLLING_INTERVAL_SECONDS", "1") or "1")),
}

# Back-compat alias for documentation and imports.
celery_app = app
