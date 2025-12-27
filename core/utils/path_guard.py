import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _test_write_access(target: Path) -> bool:
    """
    Return True if the current process can create and delete a file in target,
    including creating nested subdirectories.
    """
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe_dir = target / ".perm_probe_dir" / "nested"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe = probe_dir / ".perm_probe"
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        shutil.rmtree(probe_dir.parent, ignore_errors=True)
        return True
    except OSError:
        return False


def _tree_is_writable(root: Path, *, max_dirs: int = 5000) -> bool:
    """
    Return True if every existing directory under root is writable/executable.

    This is a defensive guard for cases where root itself is writable but contains
    child directories owned by another user (e.g., created by Docker as root).
    """
    if not root.exists():
        return True

    checked = 0

    def _onerror(_: OSError) -> None:
        raise

    try:
        for dirpath, dirnames, _ in os.walk(root, onerror=_onerror):
            checked += 1
            if checked > max_dirs:
                logger.warning("Path tree under %s is large; checked first %d directories", root, max_dirs)
                return True
            if not os.access(dirpath, os.W_OK | os.X_OK):
                return False
            # Avoid descending into dot directories created by probes or tooling
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        return True
    except OSError:
        return False


def ensure_writable_dir(
    preferred_path: str,
    fallback_path: Optional[str] = None,
) -> str:
    """
    Ensure a writable directory exists at preferred_path.

    When preferred_path is not writable, fall back to fallback_path (or
    ./local/runtime/<preferred_name>) and return the resolved location.
    """
    preferred = Path(preferred_path).expanduser().resolve()
    if _test_write_access(preferred) and _tree_is_writable(preferred):
        return str(preferred)

    runtime_root = Path(
        fallback_path
        or os.getenv("RAGARC_RUNTIME_DIR", "./local/runtime")
    ).expanduser().resolve()

    fallback = runtime_root
    if fallback_path is None:
        fallback = runtime_root / preferred.name

    if _test_write_access(fallback) and _tree_is_writable(fallback):
        logger.warning(
            "Directory %s is not writable; falling back to %s",
            preferred,
            fallback,
        )
        return str(fallback)

    raise RuntimeError(
        f"Unable to create writable directory at '{preferred}' or fallback '{fallback}'. "
        "Please adjust filesystem permissions."
    )


def require_writable_dir(path: str) -> str:
    """Ensure a writable directory exists at path, otherwise raise.

    Unlike `ensure_writable_dir`, this function does not attempt any fallback paths.
    """

    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=target, prefix=".perm_probe_", delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
    except OSError as exc:
        raise RuntimeError(f"Directory '{target}' is not writable") from exc
    if not _tree_is_writable(target):
        raise RuntimeError(f"Directory tree under '{target}' is not writable")
    return str(target)
