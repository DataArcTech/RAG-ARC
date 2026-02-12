"""Global defaults for LLM JSON output retries.

Why this lives in ``config/``:
- Avoid scattered retry constants across deepsearch/query-rewrite/extractors.
- Keep retry behavior tunable from environment variables.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning("Invalid integer for %s (%s); falling back to %s", name, raw, default)
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("Invalid float for %s (%s); falling back to %s", name, raw, default)
        return float(default)


# Total attempts for a JSON call (including the first generation).
LLM_JSON_RETRY_DEFAULT_ATTEMPTS = max(1, _env_int("LLM_JSON_RETRY_DEFAULT_ATTEMPTS", 2))
# Hard clamp to prevent accidental runaway loops.
LLM_JSON_RETRY_MAX_ATTEMPTS = max(1, _env_int("LLM_JSON_RETRY_MAX_ATTEMPTS", 8))

# Retry generation temperature for JSON repair turns.
LLM_JSON_RETRY_DEFAULT_TEMPERATURE = _env_float("LLM_JSON_RETRY_DEFAULT_TEMPERATURE", 0.0)

# Excerpt length used when feeding prior invalid output back to the model.
LLM_JSON_RETRY_DEFAULT_MAX_RAW_CHARS = max(256, _env_int("LLM_JSON_RETRY_DEFAULT_MAX_RAW_CHARS", 2000))


__all__ = [
    "LLM_JSON_RETRY_DEFAULT_ATTEMPTS",
    "LLM_JSON_RETRY_MAX_ATTEMPTS",
    "LLM_JSON_RETRY_DEFAULT_TEMPERATURE",
    "LLM_JSON_RETRY_DEFAULT_MAX_RAW_CHARS",
]
