"""Virtual path helpers.

This project uses `io://...` as a backend-agnostic virtual path scheme. In Phase 1,
`io://` paths are mapped to a LocalDB-backed directory under `IO_STORE_BASE_PATH`
(default: `./data/localdb`).
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


def localdb_root_dir() -> Path:
    """Return the absolute LocalDB root directory for io:// mapping."""

    base = str(os.getenv("IO_STORE_BASE_PATH", "./data/localdb") or "").strip() or "./data/localdb"
    path = Path(base).expanduser()
    if not path.is_absolute():
        # Use repo root as the anchor (avoid CWD-dependent resolution).
        from core.utils.filename_guard import project_root_dir

        path = (project_root_dir() / path).resolve()
    return path.resolve()


def resolve_io_to_local_path(value: str) -> Path:
    """Map an io:// path to a local filesystem path under LocalDB root."""

    key = io_key(value)
    return localdb_root_dir() / key

