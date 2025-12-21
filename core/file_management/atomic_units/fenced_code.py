import re
from typing import List, Literal, Tuple

SegmentKind = Literal["text", "code"]

_FENCE_START_RE = re.compile(r"^\s*(```|~~~)")
_FENCE_INFO_RE = re.compile(r"^\s*(```|~~~)\s*(\S+)?(?:\s+.*)?\s*$")


def is_fence_line(line: str) -> bool:
    return bool(_FENCE_START_RE.match(line or ""))


def parse_fenced_code_block(block: str) -> tuple[str, str, List[str]]:
    lines = (block or "").splitlines()
    if not lines:
        return "```", "", []

    match = _FENCE_INFO_RE.match(lines[0])
    fence = match.group(1) if match else "```"
    language = (match.group(2) or "").strip() if match else ""

    body_lines: List[str] = []
    for line in lines[1:]:
        if re.match(rf"^\s*{re.escape(fence)}\s*$", line):
            break
        body_lines.append(line)
    return fence, language, body_lines


def split_fenced_code_blocks(text: str, *, keepends: bool = True) -> List[Tuple[SegmentKind, str]]:
    """
    Split markdown-like text into segments while preserving fenced code blocks as atomic segments.

    A fenced code block starts with a line beginning with ``` or ~~~ (allowing leading whitespace)
    and ends at the first subsequent line containing only the same fence marker (allowing whitespace).
    """
    if not isinstance(text, str) or not text:
        return []

    lines = text.splitlines(keepends=True)
    segments: List[Tuple[SegmentKind, str]] = []
    buffer: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\r\n")
        match = _FENCE_START_RE.match(stripped)
        if match:
            if buffer:
                segments.append(("text", "".join(buffer)))
                buffer.clear()

            fence = match.group(1)
            start = i
            i += 1
            while i < len(lines):
                candidate = lines[i].rstrip("\r\n")
                if re.match(rf"^\s*{re.escape(fence)}\s*$", candidate):
                    i += 1
                    break
                i += 1
            segments.append(("code", "".join(lines[start:i])))
            continue

        buffer.append(line)
        i += 1

    if buffer:
        segments.append(("text", "".join(buffer)))

    def _normalize(s: str) -> str:
        if keepends:
            return s
        return s.replace("\r\n", "\n").replace("\r", "\n")

    return [(kind, _normalize(content)) for kind, content in segments if content]


def split_fenced_code_blocks_with_line_spans(text: str) -> List[Tuple[SegmentKind, int, int]]:
    """
    Split into segments like split_fenced_code_blocks, but return 0-based line spans (start, end).

    The returned spans index into `text.splitlines()` (keepends=False).
    """
    if not isinstance(text, str) or not text:
        return []

    lines = text.splitlines()
    segments: List[Tuple[SegmentKind, int, int]] = []

    i = 0
    start_text = 0
    while i < len(lines):
        line = lines[i]
        match = _FENCE_START_RE.match(line)
        if match:
            if start_text < i:
                segments.append(("text", start_text, i))

            fence = match.group(1)
            start = i
            i += 1
            while i < len(lines):
                if re.match(rf"^\s*{re.escape(fence)}\s*$", lines[i]):
                    i += 1
                    break
                i += 1
            segments.append(("code", start, i))
            start_text = i
            continue

        i += 1

    if start_text < len(lines):
        segments.append(("text", start_text, len(lines)))

    return segments

