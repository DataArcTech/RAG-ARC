"""Small helpers for extracting JSON payloads from LLM output.

This is intentionally shared (not DeepSearch-specific) so other modules can reuse the same
robust parsing logic without duplicating regex snippets.
"""
from typing import Any

from core.utils.text_regex import MARKDOWN_JSON_FENCE_RE


def extract_json_from_text(raw: str) -> str | None:
    """Extract the first JSON object/array from text (handles ```json fenced blocks)."""

    text = (raw or "").strip()
    if not text:
        return None

    fence = MARKDOWN_JSON_FENCE_RE.search(text)
    if fence:
        text = (fence.group("body") or "").strip()
        if not text:
            return None

    start_obj = text.find("{")
    start_list = text.find("[")
    if start_obj == -1 or (0 <= start_list < start_obj):
        start = start_list
    else:
        start = start_obj
    if start == -1:
        return None

    open_brace = {"{": "}", "[": "]"}
    opener = text[start]
    closer = open_brace.get(opener)
    if closer is None:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_last_json_array_from_text(raw: str) -> str | None:
    """Extract the last JSON array from text.

    This is useful for LLM outputs that may contain incidental bracketed spans (e.g. "[1]")
    before emitting the final structured array payload.
    """

    import re

    text = (raw or "").strip()
    if not text:
        return None

    fenced = list(MARKDOWN_JSON_FENCE_RE.finditer(text))
    if fenced:
        body = (fenced[-1].group("body") or "").strip()
        extracted = extract_json_from_text(body)
        if extracted and extracted.startswith("["):
            return extracted

    # Fallback: grab the last numeric JSON-like array.
    matches = re.findall(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]", text)
    for candidate in reversed(matches):
        extracted = extract_json_from_text(candidate)
        if extracted and extracted.startswith("["):
            return extracted
    return None


def safe_json_loads(raw: str, *, expected: str | None = None) -> Any | None:
    """Parse JSON from model output; returns None when parsing fails or type mismatches."""

    import json

    extracted = extract_json_from_text(raw)
    if extracted is None:
        return None
    try:
        value = json.loads(extracted)
    except json.JSONDecodeError:
        repaired = _escape_unescaped_newlines_in_json_strings(extracted)
        repaired2 = _strip_invalid_json_escapes(repaired)
        if repaired2 == extracted:
            return None
        try:
            value = json.loads(repaired2)
        except json.JSONDecodeError:
            return None
    if expected == "dict":
        return value if isinstance(value, dict) else None
    if expected == "list":
        return value if isinstance(value, list) else None
    return value


def _escape_unescaped_newlines_in_json_strings(text: str) -> str:
    """Escape literal newlines inside JSON string values (a common LLM formatting error).

    JSON does not allow raw newline characters inside quoted strings. Some models emit multi-line
    strings directly, which breaks `json.loads`. This helper converts those newlines into `\\n`
    (and `\\r`, `\\t`) only when they occur inside a quoted string literal.
    """

    if not text or "\"" not in text:
        return text

    out: list[str] = []
    in_string = False
    escape = False
    changed = False

    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == "\"":
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                changed = True
                continue
            if ch == "\r":
                out.append("\\r")
                changed = True
                continue
            if ch == "\t":
                out.append("\\t")
                changed = True
                continue
            out.append(ch)
            continue

        if ch == "\"":
            out.append(ch)
            in_string = True
            continue
        out.append(ch)

    if not changed:
        return text
    return "".join(out)


def _strip_invalid_json_escapes(text: str) -> str:
    """Remove invalid escape backslashes inside JSON string literals.

    Some LLMs emit Python-style escapes like ``\\'`` inside JSON strings. JSON only
    allows a small set of escapes: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX.
    We strip the backslash when it precedes an unsupported escape char while inside a
    quoted string.
    """

    if not text or "\\" not in text or "\"" not in text:
        return text

    allowed = {"\"", "\\", "/", "b", "f", "n", "r", "t", "u"}
    out: list[str] = []
    in_string = False
    escape = False
    changed = False

    for ch in text:
        if in_string:
            if escape:
                # We are at the char AFTER a backslash inside a string.
                if ch in allowed:
                    out.append("\\")
                    out.append(ch)
                else:
                    # Invalid JSON escape (e.g. \' or \() -> drop the backslash, keep the char.
                    out.append(ch)
                    changed = True
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            out.append(ch)
            if ch == "\"":
                in_string = False
            continue

        out.append(ch)
        if ch == "\"":
            in_string = True

    # Trailing backslash inside a string: keep it (let json.loads fail; avoids inventing chars).
    if escape:
        out.append("\\")

    if not changed:
        return text
    return "".join(out)
