"""Index path scoping helpers (owner/tenant aware).

This module centralizes the "where do we store per-owner index artifacts" policy so we
don't hardcode path conventions across FAISS/BM25/etc.
"""
import os
from typing import Iterable, List, Optional

from framework.virtual_paths import is_io_path


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
    root_virtual = os.path.join(base, str(owner_dirname or "owners"))
    if is_io_path(root_virtual):
        try:
            from framework.register import Register

            io_manager = Register().get_object("io_manager")
        except Exception:
            io_manager = None
        if io_manager is None:
            return []
        # IOManager lists object keys, not directories. Recover the immediate child folder names.
        prefix = root_virtual.rstrip("/") + "/"
        try:
            keys = io_manager.list_keys_path(root_virtual)
        except Exception:
            keys = []
        seen: set[str] = set()
        out: List[tuple[Optional[str], str]] = []
        for ref in keys:
            token = str(ref or "").strip()
            if not token.startswith(prefix):
                continue
            rel = token[len(prefix) :]
            seg = rel.split("/", 1)[0].strip()
            if not seg or seg.startswith(".") or seg in seen:
                continue
            seen.add(seg)
            owner = None if seg == global_owner_name else seg
            out.append((owner, f"{prefix}{seg}"))
        return sorted(out, key=lambda row: (row[0] is None, row[0] or ""))

    if not os.path.isdir(root_virtual):
        return []

    out: List[tuple[Optional[str], str]] = []
    for name in sorted(os.listdir(root_virtual)):
        if not name or name.startswith("."):
            continue
        full_virtual = os.path.join(root_virtual, name)
        if not os.path.isdir(full_virtual):
            continue
        owner = None if name == global_owner_name else name
        # Preserve the caller-facing directory path as `io://...` when applicable, so downstream
        # components (e.g. FAISS/BM25 loaders) can reuse the same virtual paths.
        out.append((owner, full_virtual))
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
