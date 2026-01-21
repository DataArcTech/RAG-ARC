"""Anchor variant helpers (no regex).

Anchors represent subject entities (company/product) extracted upstream. We generate
deterministic variants to match multilingual/multi-script corpora, while avoiding
over-broad ASCII-only matches for CJK anchors (e.g. "upgrade").
"""
from typing import Iterable


def contains_cjk(text: str) -> bool:
    """Coarse CJK unified ideographs check (sufficient for our safety rules)."""
    for ch in str(text or ""):
        o = ord(ch)
        if 0x3400 <= o <= 0x9FFF:
            return True
    return False


def dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        s = str(raw or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def anchor_variants(anchors: list[str]) -> list[str]:
    """Generate deterministic variants for a list of anchors."""
    variants: list[str] = []
    for a in anchors or []:
        token = str(a or "").strip()
        if not token:
            continue
        variants.append(token)
        try:
            from core.utils.query_variants import generate_query_variants

            for v in generate_query_variants(token):
                vv = str(v or "").strip()
                if not vv:
                    continue
                # If the anchor contains CJK, avoid ASCII-only variants that may be too generic.
                if contains_cjk(token) and not contains_cjk(vv):
                    continue
                variants.append(vv)
        except Exception:
            # Variant generation is an enhancement; keep it non-fatal.
            continue
    return dedupe_preserve(variants)


__all__ = ["contains_cjk", "dedupe_preserve", "anchor_variants"]

