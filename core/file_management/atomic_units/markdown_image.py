import re
from typing import List, Tuple


_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MARKDOWN_IMAGE_ALT_RE = re.compile(r"!\[([^\]]*)]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*\bsrc=(?:\"([^\"]+)\"|'([^']+)'|([^\\s>]+))[^>]*>", re.IGNORECASE)


def find_markdown_image_spans(line: str) -> List[Tuple[int, int]]:
    """Return (start,end) spans for inline markdown images in a single line."""
    if not isinstance(line, str) or not line:
        return []
    spans: List[Tuple[int, int]] = []
    for m in _MARKDOWN_IMAGE_RE.finditer(line):
        spans.append((m.start(), m.end()))
    for m in _HTML_IMG_RE.finditer(line):
        spans.append((m.start(), m.end()))
    spans.sort()
    return spans


def line_contains_image(line: str) -> bool:
    if not isinstance(line, str) or not line.strip():
        return False
    return bool(find_markdown_image_spans(line))


def extract_image_urls(line: str) -> List[str]:
    if not isinstance(line, str) or not line:
        return []
    urls: List[str] = []
    for m in _MARKDOWN_IMAGE_RE.finditer(line):
        url = (m.group(1) or "").strip()
        if url:
            urls.append(url)
    for m in _HTML_IMG_RE.finditer(line):
        url = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if url:
            urls.append(url)
    return urls


def extract_image_alts(line: str) -> List[str]:
    if not isinstance(line, str) or not line:
        return []
    alts: List[str] = []
    for m in _MARKDOWN_IMAGE_ALT_RE.finditer(line):
        alt = (m.group(1) or "").strip()
        if alt:
            alts.append(alt)
    return alts
