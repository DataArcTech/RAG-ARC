"""Helpers for cleaning user questions into retriever-friendly queries."""
import re


_PATH_LIKE_RE = re.compile(r"[/\\\\]|\\.(pdf|docx|pptx|xlsx|md|txt)\\b", flags=re.IGNORECASE)
_BULLET_RE = re.compile(r"^\\s*[-*]\\s+")
_SPACE_RE = re.compile(r"\\s+")


def clean_query(text: str, *, max_chars: int = 240) -> str:
    """Remove file lists / long metadata and return a compact query string."""

    raw = str(text or "").strip()
    if not raw:
        return ""

    kept: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("涉及文件"):
            continue
        if _BULLET_RE.match(stripped) and _PATH_LIKE_RE.search(stripped):
            continue
        kept.append(stripped)

    compact = _SPACE_RE.sub(" ", " ".join(kept)).strip()
    if not compact:
        compact = raw

    for sep in ("。", "？", "!", "?", "\n"):
        idx = compact.find(sep)
        if idx > 0:
            head = compact[: idx + 1].strip()
            if len(head) >= 8:
                compact = head
                break

    if max_chars > 0 and len(compact) > max_chars:
        compact = compact[: max_chars - 1].rstrip() + "…"
    return compact

