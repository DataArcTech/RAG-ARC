"""Index path scoping helpers (owner/tenant aware).

This module centralizes the "where do we store per-owner index artifacts" policy so we
don't hardcode path conventions across FAISS/BM25/etc.
"""
import os
from typing import Iterable, List, Optional


def _safe_token(token: str) -> str:
    # Avoid path traversal / separators; keep deterministic.
    out = str(token or "").strip() or "unknown"
    out = out.replace("/", "_").replace("\\", "_")
    return out


def owner_scoped_dir(
    base_dir: str,
    *,
    owner_id: Optional[object],
    owner_dirname: str = "owners",
    global_owner_name: str = "__GLOBAL__",
) -> str:
    base = str(base_dir or "").strip() or "."
    owner_token = global_owner_name if owner_id is None else _safe_token(str(owner_id))
    return os.path.join(base, str(owner_dirname or "owners"), owner_token)


def iter_owner_dirs(
    base_dir: str,
    *,
    owner_dirname: str = "owners",
    global_owner_name: str = "__GLOBAL__",
) -> List[tuple[Optional[str], str]]:
    """List (owner_id, dir_path) pairs found on disk under base_dir/owner_dirname."""
    base = str(base_dir or "").strip() or "."
    root = os.path.join(base, str(owner_dirname or "owners"))
    if not os.path.isdir(root):
        return []

    out: List[tuple[Optional[str], str]] = []
    for name in sorted(os.listdir(root)):
        if not name or name.startswith("."):
            continue
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        owner = None if name == global_owner_name else name
        out.append((owner, full))
    return out


def chunked(items: Iterable[str], size: int) -> List[List[str]]:
    buf: List[str] = []
    out: List[List[str]] = []
    n = max(1, int(size or 1))
    for item in items:
        buf.append(item)
        if len(buf) >= n:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out

