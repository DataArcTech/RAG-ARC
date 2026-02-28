import json
import logging
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.file_management.atomic_units.fenced_code import is_fence_line
from framework.virtual_paths import is_io_path

logger = logging.getLogger(__name__)

_PUNCT_TABLE = str.maketrans(
    "",
    "",
    string.punctuation
    + "\uFF0C\u3002\uFF1B\uFF1A\uFF01\uFF1F\uFF08\uFF09\u300A\u300B\u3010\u3011"
    + "\u201C\u201D\u2018\u2019\u00B7\u2022\u2014\u2013\u2026",
)


def normalize_heading_text(text: str) -> str:
    token = str(text or "").strip().lower()
    if not token:
        return ""
    token = token.translate(_PUNCT_TABLE)
    token = "".join(token.split())
    return token


def extract_numbering_level(text: str, *, max_level: int) -> Optional[int]:
    token = str(text or "").strip()
    if not token:
        return None
    match = re.match(r"^(\d+(?:\.\d+)*)", token)
    if not match:
        return None
    dots = match.group(1).count(".")
    level = dots + 1
    if level <= 0:
        return None
    return min(level, max_level)


def iter_markdown_headings(markdown: str) -> List[Dict[str, Any]]:
    lines = markdown.splitlines()
    headings: List[Dict[str, Any]] = []
    in_fence = False
    offset = 0
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if is_fence_line(stripped):
            in_fence = not in_fence
            offset += len(line) + 1
            continue
        if in_fence:
            offset += len(line) + 1
            continue
        if not stripped.startswith("#"):
            offset += len(line) + 1
            continue
        match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if not match:
            offset += len(line) + 1
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if not title:
            offset += len(line) + 1
            continue
        headings.append(
            {
                "title": title,
                "markdown_level": level,
                "line_index": idx,
                "char_start": offset,
                "char_end": offset + len(line),
            }
        )
        offset += len(line) + 1
    return headings


def read_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def resolve_content_list_path(md_path: Optional[str], output_dir: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if output_dir:
        token = str(output_dir).strip()
        if token and not is_io_path(token):
            candidates.extend(sorted(Path(token).glob("*_content_list*.json")))
    if md_path:
        token = str(md_path).strip()
        if token and not is_io_path(token):
            md_dir = Path(token).parent
            candidates.extend(sorted(md_dir.glob("*_content_list*.json")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_content_list_index(content_list: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for item in content_list:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        key = normalize_heading_text(text)
        if not key:
            continue
        index.setdefault(key, []).append(item)
    return index


def max_page_index(content_list: Iterable[Dict[str, Any]]) -> Optional[int]:
    values: List[int] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        if isinstance(page_idx, int):
            values.append(page_idx)
    return max(values) if values else None


def build_page_text_index(content_list: Iterable[Dict[str, Any]]) -> Dict[int, str]:
    pages: Dict[int, List[str]] = {}
    for item in content_list:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            # MinerU list-type items store content in list_items, not text.
            list_items = item.get("list_items")
            if isinstance(list_items, list) and list_items:
                text = "\n".join(str(li) for li in list_items).strip()
            else:
                # MinerU table-type items store content in table_body.
                table_body = item.get("table_body")
                if isinstance(table_body, str):
                    text = table_body.strip()
        if not text:
            continue
        pages.setdefault(page_idx, []).append(text)
    return {page: "\n".join(lines) for page, lines in pages.items()}


def normalize_for_match(text: str) -> str:
    return normalize_heading_text(text)


def estimate_tokens(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    except Exception:
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        word_tokens = len(re.findall(r"[A-Za-z0-9_]+", text))
        punct_tokens = len(re.findall(r"[^\sA-Za-z0-9_\u4e00-\u9fff]", text))
        return cjk_chars + word_tokens + punct_tokens


def truncate_text_by_tokens(text: str, *, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if not text:
        return ""
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    ratio = max_tokens / max(estimated, 1)
    char_limit = max(int(len(text) * ratio), 1)
    return text[:char_limit].rstrip()
