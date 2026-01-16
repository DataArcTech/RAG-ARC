"""Index text augmentation knobs (centralized).

These knobs control whether we inject lightweight document-title hints (derived
from `metadata.filename`) into `metadata.index_text`.

Why:
- Dense/BM25 indexing uses `index_text` when present; many chunks do not repeat
  the document/product name in every chunk body, so filename-derived titles
  improve recall for file-scoped queries.
- The same augmentation is reused for reranking inputs so existing indexes can
  benefit without requiring re-embedding.
"""

from __future__ import annotations

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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Invalid integer for %s (%s); falling back to %s", name, raw, default)
        return default


# Enable/disable augmentation globally.
INDEX_TEXT_TITLE_PREFIX_ENABLED = _env_bool("INDEX_TEXT_TITLE_PREFIX_ENABLED", True)

# Keep the injected title short to reduce prompt/index noise.
INDEX_TEXT_TITLE_MAX_CHARS = _env_int("INDEX_TEXT_TITLE_MAX_CHARS", 160)

