import re
from typing import List, Tuple

TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def is_markdown_table_start(lines: List[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    header = lines[i]
    separator = lines[i + 1]
    if "|" not in header:
        return False
    return bool(TABLE_SEPARATOR_RE.match(separator))


def parse_markdown_table(table: str) -> Tuple[str, str, List[str]]:
    lines = [line.rstrip() for line in (table or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return "", "", []
    header = lines[0]
    separator = lines[1]
    rows = lines[2:] if len(lines) > 2 else []
    return header, separator, rows


def extract_markdown_table_rows(content: str) -> List[str]:
    lines = [line.rstrip() for line in (content or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    for idx in range(len(lines) - 1):
        if "|" not in lines[idx]:
            continue
        if TABLE_SEPARATOR_RE.match(lines[idx + 1]):
            return [line for line in lines[idx + 2 :] if "|" in line]

    return [line for line in lines if "|" in line]

