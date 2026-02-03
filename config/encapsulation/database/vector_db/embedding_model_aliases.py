"""Embedding model name aliases for fingerprint normalization."""
import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

# Default aliases: normalize OpenRouter-style provider prefixes to canonical model names.
_DEFAULT_ALIASES: Dict[str, str] = {
    "openai/text-embedding-3-small": "text-embedding-3-small",
    "openai/text-embedding-3-large": "text-embedding-3-large",
}


def _load_aliases() -> Dict[str, str]:
    raw = os.getenv("EMBEDDING_MODEL_NAME_ALIASES")
    if not raw:
        return dict(_DEFAULT_ALIASES)
    try:
        payload = json.loads(raw)
    except Exception:
        logger.warning("Invalid EMBEDDING_MODEL_NAME_ALIASES; expected JSON mapping. Falling back to defaults.")
        return dict(_DEFAULT_ALIASES)
    if not isinstance(payload, dict):
        logger.warning("EMBEDDING_MODEL_NAME_ALIASES must be a JSON object; falling back to defaults.")
        return dict(_DEFAULT_ALIASES)

    cleaned: Dict[str, str] = {}
    for key, value in payload.items():
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            continue
        cleaned[k] = v
    if not cleaned:
        return dict(_DEFAULT_ALIASES)

    merged = dict(_DEFAULT_ALIASES)
    merged.update(cleaned)
    return merged


EMBEDDING_MODEL_NAME_ALIASES = _load_aliases()


def normalize_embedding_model_name(name: str | None) -> str | None:
    token = str(name or "").strip()
    if not token:
        return None
    return EMBEDDING_MODEL_NAME_ALIASES.get(token, token)


__all__ = ["EMBEDDING_MODEL_NAME_ALIASES", "normalize_embedding_model_name"]
