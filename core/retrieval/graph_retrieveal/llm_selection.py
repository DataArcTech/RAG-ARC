"""Shared LLM selection parsing helpers for graph retrieval.

Centralizes the (fragile) "pick N indices" parsing so we can make it robust and testable.
"""
import re
from typing import Any, List, Sequence

from core.utils.json_extract import safe_json_loads


def parse_ranked_choice_indices(
    raw: Any,
    *,
    candidate_count: int,
    k: int,
    one_based: bool = True,
    allow_fallback: bool = True,
) -> List[int]:
    """Return a validated list of unique indices into the candidate list."""

    n = max(int(candidate_count), 0)
    want = max(int(k), 0)
    if n <= 0 or want <= 0:
        return []

    def _normalize(values: Sequence[Any]) -> List[int]:
        indices: List[int] = []
        seen: set[int] = set()
        for item in values:
            try:
                idx = int(str(item).strip())
            except Exception:
                continue
            if one_based:
                idx -= 1
            if idx < 0 or idx >= n:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
            if len(indices) >= want:
                break
        return indices

    parsed = raw if isinstance(raw, list) else safe_json_loads(str(raw or ""), expected="list")
    if isinstance(parsed, list):
        indices = _normalize(parsed)
        if indices:
            return indices

    parsed_obj = raw if isinstance(raw, dict) else safe_json_loads(str(raw or ""), expected="dict")
    if isinstance(parsed_obj, dict):
        for key in ("indices", "selected", "choices", "ranks", "ranked"):
            value = parsed_obj.get(key)
            if isinstance(value, list):
                indices = _normalize(value)
                if indices:
                    return indices

    if not allow_fallback:
        return []

    # Backward compatible fallback: parse digits from free-form text like "1,3,5".
    candidates = re.findall(r"\d+", str(raw or ""))
    return _normalize(candidates)


__all__ = ["parse_ranked_choice_indices"]
