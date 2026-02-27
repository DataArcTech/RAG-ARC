import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _register_io_manager_for_deepsearch_tests(tmp_path_factory) -> None:
    """Ensure DeepSearch tests have a hermetic IOManager registered.

    DeepSearch components persist artifacts/traces via `io://...` virtual paths.
    Unit tests should not depend on developer-local `.env` values or write to the
    repo's default `./data/localdb` directory.
    """

    base_dir = tmp_path_factory.mktemp("ragarc_iostore")
    os.environ.setdefault("IO_STORE_BACKEND", "localdb")
    os.environ.setdefault("IO_STORE_BASE_PATH", str(base_dir))
    os.environ.setdefault("IO_STORE_DEFAULT_NAMESPACE", "io")

    from app_registration import registrator
    from config.encapsulation.io.io_manager_config import IOManagerConfig

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "json_configs" / "io_manager.json"
    registrator.register(config_path=str(config_path), app_name="io_manager", config_type=IOManagerConfig)

