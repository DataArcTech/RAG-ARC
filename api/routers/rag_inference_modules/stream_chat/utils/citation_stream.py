"""Streaming citation renumbering utilities."""
import re
from typing import Dict, List

_SUP_OPEN_PREFIX = "<sup"
_SUP_CLOSE = "</sup>"
_SUP_PREFIXES = ("<sup", "<su", "<s", "<")
_SUP_KEY_RE = re.compile(r"\d{1,4}")


class CitationStreamRenumberer:
    """Renumber <sup> citations on the fly using first-appearance order."""

    def __init__(self, *, max_key: int | None = None) -> None:
        self._buffer = ""
        self._key_map: Dict[int, int] = {}
        self._next_key = 1
        self._max_key = max_key

    @property
    def key_map(self) -> Dict[int, int]:
        """Return the current old->new citation key mapping."""
        return dict(self._key_map)

    def feed(self, text: str) -> str:
        """Process a text chunk and return renumbered output (may be empty)."""
        if not text:
            return ""
        self._buffer += text
        output: List[str] = []
        idx = 0

        while True:
            start = self._buffer.find(_SUP_OPEN_PREFIX, idx)
            if start == -1:
                tail = self._buffer[idx:]
                emit, keep = _split_partial_sup_prefix(tail)
                if emit:
                    output.append(emit)
                self._buffer = keep
                break

            if start > idx:
                output.append(self._buffer[idx:start])

            # Ensure we have the full opening tag.
            tag_end = self._buffer.find(">", start)
            if tag_end == -1:
                self._buffer = self._buffer[start:]
                break

            # Ensure we have the closing tag.
            close_start = self._buffer.find(_SUP_CLOSE, tag_end)
            if close_start == -1:
                self._buffer = self._buffer[start:]
                break

            content = self._buffer[tag_end + 1:close_start]
            output.append(self._rewrite_sup_content(content))
            idx = close_start + len(_SUP_CLOSE)

        return "".join(output)

    def flush(self) -> str:
        """Flush any buffered text (used when the stream ends)."""
        if not self._buffer:
            return ""
        output = self._buffer
        self._buffer = ""
        return output

    def set_max_key(self, max_key: int | None) -> None:
        """Set the max allowed original citation key (inclusive)."""
        self._max_key = max_key

    def _rewrite_sup_content(self, content: str) -> str:
        keys: List[int] = []
        seen = set()
        for match in _SUP_KEY_RE.finditer(content or ""):
            try:
                key = int(match.group(0))
            except ValueError:
                continue
            if key <= 0:
                continue
            if self._max_key is not None and key > int(self._max_key):
                # Ignore out-of-range citations so the streamed text stays consistent with
                # the sources list we can actually build for the frontend.
                continue
            if key not in self._key_map:
                self._key_map[key] = self._next_key
                self._next_key += 1
            new_key = self._key_map[key]
            if new_key not in seen:
                seen.add(new_key)
                keys.append(new_key)

        if not keys:
            return ""

        keys_formatted = ", ".join(str(k) for k in keys)
        return f"<sup>{keys_formatted}</sup>"


def _split_partial_sup_prefix(text: str) -> tuple[str, str]:
    """Split trailing partial <sup prefix to keep for the next chunk."""
    if not text:
        return ("", "")
    for prefix in _SUP_PREFIXES:
        if text.endswith(prefix):
            return (text[: -len(prefix)], prefix)
    return (text, "")
