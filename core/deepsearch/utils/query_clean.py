"""Helpers for cleaning user questions into retriever-friendly queries."""
import unicodedata

from core.utils.text_regex import BULLET_RE as _BULLET_RE
from core.utils.text_regex import PATH_LIKE_RE as _PATH_LIKE_RE
from core.utils.text_regex import WHITESPACE_RE as _SPACE_RE


def clean_query(text: str, *, max_chars: int = 240) -> str:
    """Remove file lists / long metadata and return a compact query string."""

    raw = str(text or "").strip()
    if not raw:
        return ""

    kept: list[str] = []
    lines = raw.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Drop generic "header" lines that introduce a subsequent list of file paths, without
        # relying on any specific language tokens (e.g. "Files:", "涉及文件：", etc.).
        if stripped.endswith((":","：")):
            look = idx + 1
            while look < len(lines) and not lines[look].strip():
                look += 1
            if look < len(lines):
                nxt = lines[look].strip()
                if _BULLET_RE.match(nxt) and _PATH_LIKE_RE.search(nxt):
                    continue
        if _BULLET_RE.match(stripped) and _PATH_LIKE_RE.search(stripped):
            continue
        kept.append(stripped)

    compact = _SPACE_RE.sub(" ", " ".join(kept)).strip()
    if not compact:
        compact = raw

    def _is_sentence_terminator(ch: str) -> bool:
        if ch in ".!?":
            return True
        name = unicodedata.name(ch, "")
        return ("FULL STOP" in name) or ("QUESTION MARK" in name) or ("EXCLAMATION MARK" in name)

    for idx, ch in enumerate(compact):
        if ch == "\n":
            head = compact[: idx].strip()
            if len(head) >= 8:
                compact = head
                break
            continue
        if _is_sentence_terminator(ch):
            head = compact[: idx + 1].strip()
            if len(head) >= 8:
                compact = head
                break

    if max_chars > 0 and len(compact) > max_chars:
        compact = compact[: max_chars - 1].rstrip() + "…"
    return compact
