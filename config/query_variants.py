"""Query variant generation knobs (centralized).

These knobs control whether the retrieval layer should generate lightweight
query variants (e.g., Simplified/Traditional Chinese) and union results.

Rationale (general-domain):
- In mixed corpora, document text may be Traditional Chinese while users ask in
  Simplified Chinese (and vice versa). Deterministic variants improve recall
  without domain-specific rules.
"""
import logging
import os

logger = logging.getLogger(__name__)

_ALLOWED_LANGS = {"zh-hans", "zh-hant", "en"}


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


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    items = [p.strip() for p in raw.split(",")]
    items = [p for p in items if p]
    return items if items else list(default)


# Enable query variant generation globally (retrieval-time; affects Dense/BM25/HippoRAG).
QUERY_VARIANTS_ENABLED = _env_bool("QUERY_VARIANTS_ENABLED", True)

# Which variant "languages" to attempt, in order. This is intentionally lightweight:
# - zh-Hans/zh-Hant are deterministic OpenCC conversions (when available).
# - en is a best-effort ASCII token extraction (no LLM translation in core).
#
# Default order is chosen to match typical user input preference:
#   Simplified Chinese -> English tokens -> Traditional Chinese
QUERY_VARIANTS_LANGS = _env_csv("QUERY_VARIANTS_LANGS", ["zh-Hans", "en", "zh-Hant"])

# Backward-compatibility knob: if users disable Hans/Hant variants explicitly, drop them from LANGS.
QUERY_VARIANTS_ZH_HANS_HANT_ENABLED = _env_bool("QUERY_VARIANTS_ZH_HANS_HANT_ENABLED", True)
if not QUERY_VARIANTS_ZH_HANS_HANT_ENABLED:
    QUERY_VARIANTS_LANGS = [x for x in QUERY_VARIANTS_LANGS if str(x).strip().lower() not in {"zh-hans", "zh-hant"}]

# Validate + normalize LANGS.
_normalized_langs: list[str] = []
for item in QUERY_VARIANTS_LANGS:
    token = str(item or "").strip()
    if not token:
        continue
    key = token.lower()
    if key not in _ALLOWED_LANGS:
        logger.warning("Ignoring unsupported QUERY_VARIANTS_LANGS entry: %r (allowed=%s)", token, sorted(_ALLOWED_LANGS))
        continue
    # Preserve canonical casing for readability in logs/debug.
    canonical = {"zh-hans": "zh-Hans", "zh-hant": "zh-Hant", "en": "en"}[key]
    if canonical not in _normalized_langs:
        _normalized_langs.append(canonical)
QUERY_VARIANTS_LANGS = _normalized_langs

# Hard cap for number of unique variants (including the original query).
QUERY_VARIANTS_MAX = max(1, _env_int("QUERY_VARIANTS_MAX", 3))
