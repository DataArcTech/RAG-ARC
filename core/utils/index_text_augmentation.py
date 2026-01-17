"""Helpers for augmenting chunk `index_text` with filename-derived titles.

This is intentionally domain-agnostic: it improves retrieval/rerank for any
dataset where the document title/product name appears in the filename/path but
not in every chunk body.
"""
from pathlib import Path
from typing import Optional

from config.index_text_augmentation import INDEX_TEXT_TITLE_MAX_CHARS, INDEX_TEXT_TITLE_PREFIX_ENABLED


def _coerce_str(value: object) -> str:
    return str(value or "").strip()


def extract_title_from_filename(filename: str) -> str:
    """Extract a stable title token from an on-disk filename/path."""
    raw = _coerce_str(filename)
    if not raw:
        return ""
    # `filename` is typically repo-relative (e.g., "RAG-ARC/local/user_files/.../x.pdf").
    # We only want the basename stem as a portable title hint.
    title = Path(raw).name
    title = Path(title).stem  # drop extension
    return _coerce_str(title)


def build_title_prefix(filename: str) -> str:
    """Build the injected prefix line, respecting length limits."""
    title = extract_title_from_filename(filename)
    if not title:
        return ""
    max_chars = int(INDEX_TEXT_TITLE_MAX_CHARS)
    if max_chars > 0 and len(title) > max_chars:
        title = title[:max_chars].rstrip()
    # Keep the prefix format stable and language-neutral.
    return f"title={title}"


def prepend_title_prefix(*, text: str, filename: Optional[str]) -> str:
    """Prepend a filename-derived title hint to index/rerank text."""
    if not INDEX_TEXT_TITLE_PREFIX_ENABLED:
        return text or ""
    base = _coerce_str(text)
    fn = _coerce_str(filename or "")
    prefix = build_title_prefix(fn)
    if not prefix:
        return base

    # Avoid duplicating the prefix when chunk text already includes it.
    head = base.lstrip()
    if head.startswith("title=") or head.startswith(prefix):
        return base

    return f"{prefix}\n{base}".strip()

