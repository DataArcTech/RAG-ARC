import os

from framework.runtime_warnings import configure_runtime_warnings


def pytest_configure() -> None:
    """
    Keep unit tests hermetic.

    Many modules in this repo call `dotenv.load_dotenv()` (directly or indirectly),
    which can pull developer-local `.env` settings into the test process and make
    the suite depend on the workstation configuration.

    Default the task-queue mode to `inprocess` unless a test explicitly overrides
    it (e.g. celery-mode e2e tests via `monkeypatch.setenv`).
    """

    configure_runtime_warnings()
    os.environ.setdefault("TASK_QUEUE_MODE", "inprocess")
    os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
    # Knowledge indexing preflight checks attempt to connect to Postgres/Redis/Neo4j.
    # Unit tests are hermetic by default and should not require external services.
    os.environ.setdefault("RAGARC_INDEXING_DEPENDENCY_CHECK_MODE", "off")
    # Avoid filesystem/object-store probes in unit tests unless explicitly enabled.
    os.environ.setdefault("KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS", "0")
