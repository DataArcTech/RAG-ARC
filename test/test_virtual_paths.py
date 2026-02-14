from core.utils.filename_guard import project_root_dir
from framework.virtual_paths import resolve_io_to_local_path


def test_resolve_io_to_local_path_localdb_maps_to_store_root(monkeypatch) -> None:
    monkeypatch.setenv("IO_STORE_BACKEND", "localdb")
    monkeypatch.setenv("IO_STORE_BASE_PATH", "./data/localdb")
    monkeypatch.setenv("IO_STORE_STAGE_BASE_PATH", "./data/io_stage")

    path = resolve_io_to_local_path("io://parsed_files/doc/a.txt")
    assert path == (project_root_dir() / "data/localdb/parsed_files/doc/a.txt").resolve()


def test_resolve_io_to_local_path_minio_defaults_to_stage(monkeypatch) -> None:
    monkeypatch.setenv("IO_STORE_BACKEND", "minio")
    monkeypatch.setenv("IO_STORE_BASE_PATH", "./data/localdb")
    monkeypatch.setenv("IO_STORE_STAGE_BASE_PATH", "./data/io_stage")
    monkeypatch.delenv("IO_LOCAL_PERSIST_NAMESPACES", raising=False)

    import pytest

    with pytest.raises(RuntimeError):
        resolve_io_to_local_path("io://parsed_files/doc/a.txt")


def test_resolve_io_to_local_path_minio_routes_index_namespaces_to_localdb(monkeypatch) -> None:
    monkeypatch.setenv("IO_STORE_BACKEND", "minio")
    monkeypatch.setenv("IO_STORE_BASE_PATH", "./data/localdb")
    monkeypatch.setenv("IO_STORE_STAGE_BASE_PATH", "./data/io_stage")
    monkeypatch.delenv("IO_LOCAL_PERSIST_NAMESPACES", raising=False)

    faiss_path = resolve_io_to_local_path("io://unified_faiss_index/owners/o1/index.faiss")
    assert faiss_path == (project_root_dir() / "data/localdb/unified_faiss_index/owners/o1/index.faiss").resolve()

    bm25_path = resolve_io_to_local_path("io://unified_bm25_index/owners/o1/index.pkl")
    assert bm25_path == (project_root_dir() / "data/localdb/unified_bm25_index/owners/o1/index.pkl").resolve()

    graph_path = resolve_io_to_local_path("io://graph_index_neo4j/embedding_cache/cache.json")
    assert graph_path == (project_root_dir() / "data/localdb/graph_index_neo4j/embedding_cache/cache.json").resolve()


def test_resolve_io_to_local_path_minio_local_persist_namespaces_override(monkeypatch) -> None:
    monkeypatch.setenv("IO_STORE_BACKEND", "minio")
    monkeypatch.setenv("IO_STORE_BASE_PATH", "./data/localdb")
    monkeypatch.setenv("IO_STORE_STAGE_BASE_PATH", "./data/io_stage")
    monkeypatch.setenv("IO_LOCAL_PERSIST_NAMESPACES", "runtime,other_ns")

    path = resolve_io_to_local_path("io://runtime/jwt_secret_key")
    assert path == (project_root_dir() / "data/localdb/runtime/jwt_secret_key").resolve()
