from application.knowledge import celery_tasks


class _FakeRedis:
    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._expirations: dict[str, int] = {}

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):  # noqa: ARG002
        if nx and key in self._strings:
            return False
        self._strings[key] = value
        if ex is not None:
            self._expirations[key] = int(ex)
        return True

    def get(self, key: str):
        return self._strings.get(key)

    def expire(self, key: str, seconds: int):  # noqa: ARG002
        if key not in self._strings:
            return False
        self._expirations[key] = int(seconds)
        return True

    def eval(self, script: str, numkeys: int, key: str, token: str):  # noqa: ARG002
        if self._strings.get(key) == token:
            self._strings.pop(key, None)
            self._expirations.pop(key, None)
            return 1
        return 0


def test_knowledge_celery_tasks_share_single_file_lock():
    fake = _FakeRedis()
    lock_key = celery_tasks._file_lock_key(namespace="test:mq", file_id="file_1")  # noqa: SLF001

    assert celery_tasks._acquire_lock(fake, lock_key, "run_a", ttl_seconds=60) is True  # noqa: SLF001
    assert celery_tasks._acquire_lock(fake, lock_key, "run_b", ttl_seconds=60) is False  # noqa: SLF001
    assert celery_tasks._acquire_lock(fake, lock_key, "run_a", ttl_seconds=60) is True  # noqa: SLF001
    celery_tasks._release_lock(fake, lock_key, "run_a")  # noqa: SLF001
    assert celery_tasks._acquire_lock(fake, lock_key, "run_b", ttl_seconds=60) is True  # noqa: SLF001
