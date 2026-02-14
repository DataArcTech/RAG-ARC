"""Virtual path helpers.

This project uses `io://...` as a backend-agnostic virtual path scheme.

Filesystem mapping notes:
- When `IO_STORE_BACKEND=localdb`, `io://...` maps to the LocalDB root under `IO_STORE_BASE_PATH`
  (default: `./data/localdb`).
- When `IO_STORE_BACKEND=minio`, local filesystem mapping is disabled by default to avoid
  creating any local files. Only a small set of namespaces (e.g., index directories) remain
  local-persistent under `IO_STORE_BASE_PATH` (see `IO_LOCAL_PERSIST_NAMESPACES`).

The intent is that in MinIO mode, the *source of truth* lives in object storage; the
system should not rely on local disk for non-index namespaces.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


IO_PATH_PREFIX = "io://"


def is_io_path(value: object) -> bool:
    return isinstance(value, str) and value.strip().startswith(IO_PATH_PREFIX)


def io_key(value: str) -> str:
    """Extract the normalized key portion from an `io://...` path."""

    text = str(value or "").strip()
    if not text.startswith(IO_PATH_PREFIX):
        raise ValueError(f"expected an io:// path, got: {value!r}")
    raw = text[len(IO_PATH_PREFIX) :].strip().replace("\\", "/").lstrip("/")
    parts = [p for p in PurePosixPath(raw).parts if p not in {"", ".", ".."}]
    return "/".join(parts)


def io_namespace(value: str) -> str:
    """Extract the top-level namespace from an `io://...` path."""

    key = io_key(value)
    head = key.split("/", 1)[0] if key else ""
    return head.strip()


def _resolve_root_dir(base: str) -> Path:
    path = Path(base).expanduser()
    if not path.is_absolute():
        # Use repo root as the anchor (avoid CWD-dependent resolution).
        from core.utils.filename_guard import project_root_dir

        path = (project_root_dir() / path).resolve()
    return path.resolve()


def localdb_root_dir() -> Path:
    """Return the persistent LocalDB filesystem root (`IO_STORE_BASE_PATH`)."""

    base = str(os.getenv("IO_STORE_BASE_PATH", "./data/localdb") or "").strip() or "./data/localdb"
    return _resolve_root_dir(base)


def io_stage_root_dir() -> Path:
    """Return the staging/spool filesystem root (`IO_STORE_STAGE_BASE_PATH`).

    NOTE: Staging is not used when `IO_STORE_BACKEND=minio` under the "no local files" policy.
    Kept as a named path for potential future opt-in workflows.
    """

    base = str(os.getenv("IO_STORE_STAGE_BASE_PATH", "") or "").strip()
    if not base:
        raise RuntimeError("IO_STORE_STAGE_BASE_PATH is not set")
    return _resolve_root_dir(base)


def _local_persist_namespaces() -> set[str]:
    raw = str(
        os.getenv(
            "IO_LOCAL_PERSIST_NAMESPACES",
            ",".join(
                [
                    "unified_faiss_index",
                    "section_faiss_index",
                    "unified_bm25_index",
                    "section_bm25_index",
                    "graph_index_neo4j",
                    "graph_index",
                ]
            ),
        )
        or ""
    )
    return {p.strip().strip("/").replace("\\", "/") for p in raw.split(",") if p.strip()}


def io_filesystem_root_dir(value: str) -> Path:
    """Return the filesystem root directory to use for a given `io://...` path.

    - `IO_STORE_BACKEND=localdb`: always uses the persistent LocalDB root.
    - `IO_STORE_BACKEND=minio`: routes selected local-persist namespaces (e.g. index dirs) to
      the persistent LocalDB root; all other namespaces have no local mapping.
    """

    backend = str(os.getenv("IO_STORE_BACKEND", "localdb") or "localdb").strip().lower() or "localdb"
    if backend != "minio":
        return localdb_root_dir()

    ns = io_namespace(value)
    if ns and ns in _local_persist_namespaces():
        return localdb_root_dir()
    raise RuntimeError(
        f"Local filesystem mapping for {value!r} is disabled when IO_STORE_BACKEND=minio "
        f"(namespace={ns!r}). Use IOManager to read/write io:// objects."
    )


def resolve_io_to_local_path(value: str) -> Path:
    """Map an io:// path to a local filesystem path under LocalDB root."""

    key = io_key(value)
    return io_filesystem_root_dir(value) / key
