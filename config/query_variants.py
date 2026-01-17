"""Query variant generation knobs (centralized).

These knobs control whether the retrieval layer should generate lightweight
query variants (e.g., Simplified/Traditional Chinese) and union results.

Rationale (general-domain):
- In mixed corpora, document text may be Traditional Chinese while users ask in
  Simplified Chinese (and vice versa). Deterministic variants improve recall
  without domain-specific rules.
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


# Enable query variant generation globally (retrieval-time; affects Dense/BM25/HippoRAG).
QUERY_VARIANTS_ENABLED = _env_bool("QUERY_VARIANTS_ENABLED", True)

# Enable Hans<->Hant conversion variants (requires `opencc` runtime).
QUERY_VARIANTS_ZH_HANS_HANT_ENABLED = _env_bool("QUERY_VARIANTS_ZH_HANS_HANT_ENABLED", True)

# Hard cap for number of unique variants (including the original query).
QUERY_VARIANTS_MAX = max(1, _env_int("QUERY_VARIANTS_MAX", 3))

