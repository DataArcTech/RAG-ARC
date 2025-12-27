import os


def pytest_configure() -> None:
    """
    Keep unit tests hermetic.

    Many modules in this repo call `dotenv.load_dotenv()` (directly or indirectly),
    which can pull developer-local `.env` settings into the test process and make
    the suite depend on the workstation configuration.

    Default the task-queue mode to `inprocess` unless a test explicitly overrides
    it (e.g. celery-mode e2e tests via `monkeypatch.setenv`).
    """

    os.environ.setdefault("TASK_QUEUE_MODE", "inprocess")
    os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
