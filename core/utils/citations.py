import re
from typing import List, Tuple

# Strict numeric citations: we only treat <sup>123</sup> as a citation key.
_SUP_RE = re.compile(r"<sup>\s*(?P<key>\d{1,4})\s*</sup>")


def extract_sup_keys(text: str, *, max_key: int | None = None) -> List[int]:
    """
    Extract citation keys from HTML <sup> tags.

    - Only matches <sup>NUMBER</sup> (strict digits).
    - If max_key is provided, filters to 1..max_key.
    - Keeps appearance order (may contain duplicates).
    """
    out: List[int] = []
    for m in _SUP_RE.finditer(str(text or "")):
        try:
            key = int(m.group("key"))
        except Exception:  # noqa: BLE001
            continue
        if key <= 0:
            continue
        if max_key is not None and key > int(max_key):
            continue
        out.append(key)
    return out


def _normalize_adjacent_sup_groups(text: str) -> str:
    """
    Normalize adjacent citation groups like:
      <sup>2</sup> <sup>1</sup>  ->  <sup>1</sup> <sup>2</sup>

    This makes citations stable for downstream parsing/UI.
    """
    # Allow a small set of common separators between citations.
    # Use unicode escapes to keep source ASCII-only.
    seps = ",;/" + "\uFF0C\uFF1B\u3001"
    sep_re = re.escape(seps)

    group_re = re.compile(
        r"(?P<group>(?:<sup>\s*\d{1,4}\s*</sup>(?:[\s" + sep_re + r"]*)?){2,})"
    )

    def _repl(match: re.Match) -> str:
        group = match.group("group") or ""
        keys: List[int] = []
        for k in _SUP_RE.findall(group):
            try:
                keys.append(int(k))
            except Exception:  # noqa: BLE001
                continue
        if not keys:
            return group
        uniq_sorted = sorted(set(keys))
        # Join without spaces to satisfy DeepSearch citation rules.
        return "".join(f"<sup>{k}</sup>" for k in uniq_sorted)

    return group_re.sub(_repl, str(text or ""))


def compact_sup_citations(text: str, *, max_key: int) -> Tuple[str, List[int]]:
    """
    Compact citations to a dense 1..N sequence and return the original keys used.

    Example:
      text: "... <sup>1</sup> <sup>3</sup> <sup>6</sup> ..."
      max_key: 10
      -> new text uses <sup>1</sup> <sup>2</sup> <sup>3</sup>
      -> returns [1, 3, 6] (original keys, ascending)

    Notes:
    - Only keys in 1..max_key are considered for compaction.
    - After compaction, adjacent citation groups are sorted and de-duplicated.
    """
    raw = str(text or "")
    cited = set(extract_sup_keys(raw, max_key=max_key))
    if not cited:
        return raw, []

    used_keys = sorted(cited)
    remap = {old: i + 1 for i, old in enumerate(used_keys)}

    def _replace(match: re.Match) -> str:
        try:
            old = int(match.group("key"))
        except Exception:  # noqa: BLE001
            return match.group(0)
        new = remap.get(old)
        if not new:
            # Unknown/out-of-range citations: keep as-is (makes issues observable).
            return match.group(0)
        return f"<sup>{new}</sup>"

    rewritten = _SUP_RE.sub(_replace, raw)
    rewritten = _normalize_adjacent_sup_groups(rewritten)
    return rewritten, used_keys
