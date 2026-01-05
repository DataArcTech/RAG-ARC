import uuid

import pytest


def test_redis_task_queue_fails_fast_in_celery_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_QUEUE_MODE", "celery")
    monkeypatch.setenv("MQ_NAMESPACE", f"test:mq:{uuid.uuid4().hex}")

    import encapsulation.message_queue.redis_task_queue as redis_task_queue_module

    class _BoomRedisDB:  # noqa: D401
        def __init__(self, *args, **kwargs):  # noqa: ANN001, ARG002
            raise RuntimeError("redis down")

    monkeypatch.setattr(redis_task_queue_module, "RedisDB", _BoomRedisDB, raising=True)

    from encapsulation.message_queue.redis_task_queue import RedisTaskQueue

    queue = RedisTaskQueue.from_env()
    with pytest.raises(RuntimeError, match="RedisTaskQueue unavailable"):
        queue.get_task_run("any")

