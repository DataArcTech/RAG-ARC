import os
import sys
from pathlib import Path

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
