"""Deterministic query variants for retrieval (domain-agnostic).

Currently supported:
- Simplified/Traditional Chinese (Hans<->Hant) via OpenCC, when enabled.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from config.query_variants import QUERY_VARIANTS_ENABLED, QUERY_VARIANTS_MAX, QUERY_VARIANTS_ZH_HANS_HANT_ENABLED

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


@lru_cache(maxsize=4)
def _opencc_converter(name: str):
    # Lazy import to keep core import surface small and allow deployments without OpenCC.
    import opencc  # type: ignore[import-not-found]

    return opencc.OpenCC(name)


def _try_opencc_convert(query: str, *, converter: str) -> str | None:
    try:
        cc = _opencc_converter(converter)
        out = cc.convert(query)
        out = str(out or "").strip()
        return out or None
    except Exception as exc:  # noqa: BLE001
        # Keep it observable but non-fatal: variant generation is an enhancement.
        logger.debug("OpenCC convert failed (converter=%s): %s", converter, exc)
        return None


def generate_query_variants(query: str) -> list[str]:
    """
    Generate a small set of deterministic query variants.

    Always includes the original query (first).
    """
    base = str(query or "").strip()
    if not base:
        return []

    if not QUERY_VARIANTS_ENABLED:
        return [base]

    variants: list[str] = [base]

    if QUERY_VARIANTS_ZH_HANS_HANT_ENABLED:
        # s2t: Simplified -> Traditional; t2s: Traditional -> Simplified
        s2t = _try_opencc_convert(base, converter="s2t")
        t2s = _try_opencc_convert(base, converter="t2s")
        if s2t:
            variants.append(s2t)
        if t2s:
            variants.append(t2s)

    variants = _dedupe_preserve_order(variants)
    return variants[: int(QUERY_VARIANTS_MAX)]


__all__ = ["generate_query_variants"]
