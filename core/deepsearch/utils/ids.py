"""Identifier helpers for DeepSearch tools.

Why:
- Tool calls often pass identifiers sourced from OCR/markdown (e.g. image hashes like
  `images/<sha256>.jpg`). Models can accidentally treat those as `file_id`.
- DeepSearch file-level scoping must be stable and explicit; we only accept UUID file_id
  in tool args (no filename/hash fallback).

This module avoids regex-based validation and uses UUID parsing instead.
"""
import uuid
from typing import Any, Iterable, Tuple, List


def normalize_uuid(value: Any) -> str | None:
    """Return canonical UUID string when parseable, otherwise None."""

    token = str(value or "").strip()
    if not token:
        return None
    try:
        return str(uuid.UUID(token))
    except Exception:
        return None


def coerce_uuid_list(values: Iterable[Any]) -> Tuple[List[str], List[str]]:
    """Return (valid_uuids, invalid_tokens) preserving first-seen order."""

    valid: List[str] = []
    invalid: List[str] = []
    seen: set[str] = set()
    for item in values:
        raw = str(item or "").strip()
        if not raw:
            continue
        normalized = normalize_uuid(raw)
        if not normalized:
            invalid.append(raw)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        valid.append(normalized)
    return valid, invalid

