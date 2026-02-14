import uuid

from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.io.io_manager_config import IOManagerConfig


def test_io_manager_put_get_roundtrip(tmp_path):
    cfg = IOManagerConfig(
        file_db_config=LocalDBConfig(base_path=str(tmp_path)),
        default_namespace="io",
    )
    io = cfg.build()

    key = f"runs/{uuid.uuid4().hex}/payload.json"
    payload = {"ok": True, "n": 1}
    put = io.put_json(namespace="deepsearch_traces", key=key, payload=payload)
    assert put.ref.startswith("io://")

    loaded = io.get_json(put.ref)
    assert loaded == payload


def test_io_manager_normalizes_traversal_tokens(tmp_path):
    cfg = IOManagerConfig(
        file_db_config=LocalDBConfig(base_path=str(tmp_path)),
        default_namespace="io",
    )
    io = cfg.build()

    put = io.put_text(namespace="ns", key="../evil.txt", text="x")
    assert put.ref.startswith("io://")
    assert ".." not in put.key
    assert io.get_text(put.ref) == "x"

