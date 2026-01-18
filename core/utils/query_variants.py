"""Deterministic query variants for retrieval (domain-agnostic).

Currently supported:
- Simplified/Traditional Chinese (Hans<->Hant) via OpenCC, when available.
- English token variant: best-effort ASCII token extraction (no translation).
"""
import logging
import re
from functools import lru_cache

from config.query_variants import (
    QUERY_VARIANTS_ENABLED,
    QUERY_VARIANTS_LANGS,
    QUERY_VARIANTS_MAX,
    QUERY_VARIANTS_ZH_HANS_HANT_ENABLED,
)

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


_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


def _extract_ascii_tokens(query: str) -> str | None:
    tokens = _ASCII_TOKEN_RE.findall(str(query or ""))
    tokens = [t.strip() for t in tokens if t and t.strip()]
    if not tokens:
        return None
    out = " ".join(tokens).strip()
    return out or None


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

    # Add variants in configured language order, but always keep the original query first.
    for lang in QUERY_VARIANTS_LANGS:
        key = str(lang or "").strip().lower()
        if not key:
            continue

        if key in {"zh-hans", "zh-hant"}:
            if not QUERY_VARIANTS_ZH_HANS_HANT_ENABLED:
                continue
            # For robustness we don't attempt to detect the input script:
            # - t2s normalizes to Simplified
            # - s2t normalizes to Traditional
            converter = "t2s" if key == "zh-hans" else "s2t"
            out = _try_opencc_convert(base, converter=converter)
            if out:
                variants.append(out)
            continue

        if key == "en":
            out = _extract_ascii_tokens(base)
            if out:
                variants.append(out)
            continue

    variants = _dedupe_preserve_order(variants)
    return variants[: int(QUERY_VARIANTS_MAX)]


__all__ = ["generate_query_variants"]
