"""Runtime knobs for knowledge visibility checks (centralized).

These are used by `Knowledge.is_file_active` and retrieval-time filtering.
"""
import logging
import os

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean for %s (%s); falling back to %s", name, raw, default)
    return default


# Whether `Knowledge.is_file_active` should also verify the underlying blob exists.
# Default True for production safety (prevents returning citations that can't be downloaded).
KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS = _env_bool("KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS", True)

