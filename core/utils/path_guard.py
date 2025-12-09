import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _test_write_access(target: Path) -> bool:
    """
    Return True if the current process can create and delete a file in target.
    """
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".perm_probe"
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        probe.unlink()
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
    if _test_write_access(preferred):
        return str(preferred)

    runtime_root = Path(
        fallback_path
        or os.getenv("RAGARC_RUNTIME_DIR", "./local/runtime")
    ).expanduser().resolve()

    fallback = runtime_root
    if fallback_path is None:
        fallback = runtime_root / preferred.name

    if _test_write_access(fallback):
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
