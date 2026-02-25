import os
import sys
from pathlib import Path

import pytest

from framework.runtime_warnings import configure_runtime_warnings


def pytest_addoption(parser) -> None:  # noqa: ANN001
    """Provide a lightweight `--timeout` flag for hermetic CI runs.

    The upstream `pytest-timeout` plugin is not a hard dependency for this repo, but
    some verification commands use `--timeout` to guard against hangs.
    """

    parser.addoption(
        "--timeout",
        action="store",
        default=None,
        help="Per-test timeout in seconds (best-effort; implemented via SIGALRM on Unix).",
    )


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
    # Ensure the repo root is importable even when pytest adds `test/` ahead of it.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # IMPORTANT: unit tests must not depend on developer-local `.env` values.
    # Force in-process task queue mode unless a test explicitly overrides it via monkeypatch.
    os.environ["TASK_QUEUE_MODE"] = "inprocess"
    os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
    # Knowledge indexing preflight checks attempt to connect to Postgres/Redis/Neo4j.
    # Unit tests are hermetic by default and should not require external services.
    os.environ.setdefault("RAGARC_INDEXING_DEPENDENCY_CHECK_MODE", "off")
    # Avoid filesystem/object-store probes in unit tests unless explicitly enabled.
    os.environ.setdefault("KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS", "0")
    # PageIndex section summaries use an LLM and must never run in hermetic unit tests.
    # Disable them even if the host environment has API keys configured.
    os.environ["SECTION_SUMMARY_ENABLED"] = "0"
    # Ensure unit tests do not accidentally use host-provided API keys.
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("CHAT_API_KEY", None)


@pytest.fixture(scope="session", autouse=True)
def _register_io_manager_for_tests(tmp_path_factory) -> None:
    """Ensure unit tests have a hermetic IOManager registered.

    Many components persist artifacts via `io://...` virtual paths. Unit tests should
    not depend on developer-local `.env` values or write into the repo's default
    `./data/localdb` directory unless explicitly requested by a test.
    """

    base_dir = tmp_path_factory.mktemp("ragarc_iostore")
    os.environ.setdefault("IO_STORE_BACKEND", "localdb")
    os.environ.setdefault("IO_STORE_BASE_PATH", str(base_dir))
    os.environ.setdefault("IO_STORE_DEFAULT_NAMESPACE", "io")

    from app_registration import registrator
    from config.encapsulation.io.io_manager_config import IOManagerConfig

    # Avoid re-registering if a test suite already initialized IOManager.
    try:
        existing = registrator.get_object("io_manager")
    except Exception:
        existing = None
    if existing is not None:
        return

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config" / "json_configs" / "io_manager.json"
    registrator.register(config_path=str(config_path), app_name="io_manager", config_type=IOManagerConfig)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # noqa: ANN001,ARG001
    timeout_raw = item.config.getoption("timeout")
    try:
        timeout_s = int(timeout_raw) if timeout_raw is not None else 0
    except Exception:
        timeout_s = 0

    if timeout_s <= 0:
        yield
        return

    import signal

    class _TestTimeout(Exception):
        pass

    def _handler(signum, frame):  # noqa: ARG001
        raise _TestTimeout(f"Test exceeded timeout={timeout_s}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def pytest_runtest_setup(item) -> None:  # noqa: ANN001
    # Clear in-process caches between tests to avoid cross-test coupling.
    try:
        from config.core.retrieval.dense_config import DenseRetrieverConfig

        DenseRetrieverConfig.clear_process_cache()
    except Exception:
        pass
    try:
        from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig

        TantivyBM25RetrieverConfig.clear_process_cache()
    except Exception:
        pass
    try:
        from config.core.retrieval.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jRetrievalConfig

        PrunedHippoRAGNeo4jRetrievalConfig.clear_process_cache()
    except Exception:
        pass
    try:
        from encapsulation.llm.utils.openai_client import clear_openai_client_cache

        clear_openai_client_cache()
    except Exception:
        pass
