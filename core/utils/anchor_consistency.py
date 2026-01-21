"""Helpers to keep anchors self-consistent with rewritten queries (no regex)."""
from core.utils.anchor_variants import anchor_variants, dedupe_preserve
from core.utils.text_contains import contains_any


def prune_anchors_by_query_text(*, anchors: list[str] | None, rewritten_query: str | None) -> list[str]:
    """
    Prune anchors that do not appear in the rewritten query (directly or via deterministic variants).

    Motivation:
    - In CORRECTION turns, users often mention both the intended subject and the mistaken subject.
      If anchors contain both, downstream evidence filtering may keep both file sets and reintroduce drift.
    - This is domain-agnostic and does not rely on hardcoded keyword lists.

    Safety:
    - If pruning would drop everything, return the original anchors unchanged.
    """
    src = dedupe_preserve(list(anchors or []))
    q = str(rewritten_query or "").strip()
    if not src or not q:
        return src

    kept: list[str] = []
    for a in src:
        variants = anchor_variants([a])
        if variants and contains_any(q, variants):
            kept.append(a)
    return kept or src


__all__ = ["prune_anchors_by_query_text"]

