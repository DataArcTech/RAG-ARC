"""Small helpers for substring checks without regex.

We intentionally avoid regex in product logic to reduce brittleness across domains.
"""
def _norm(s: str) -> str:
    # casefold handles more Unicode cases than lower().
    return str(s or "").casefold()


def contains_any(haystack: str, needles: list[str] | tuple[str, ...] | set[str]) -> bool:
    h = _norm(haystack)
    if not h:
        return False
    for n in needles:
        token = _norm(str(n or ""))
        if not token:
            continue
        if token in h:
            return True
    return False


__all__ = ["contains_any"]

