"""Identifier helpers for DeepSearch tools.

Why:
- Tool calls often pass identifiers sourced from OCR/markdown (e.g. image hashes like
  `images/<sha256>.jpg`). Models can accidentally treat those as `file_id`.
- DeepSearch file-level scoping must be stable and explicit; we only accept UUID file_id
  in tool args (no filename/hash fallback).

This module avoids regex-based validation and uses UUID parsing instead.
"""
import json
import os
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


def _basename(value: str) -> str:
    """Extract filename from a path (handles both / and \\)."""
    return os.path.basename(value.replace("\\", "/")).strip()


async def resolve_file_ref(
    extra: dict[str, Any],
    *,
    adapter: Any = None,
    access_scope: Any = None,
) -> tuple[str | None, str | None]:
    """Resolve file_id/file/filename from tool args to (normalized_uuid, raw_value).

    Priority: file_id > file > filename.
    If the value is a UUID, return it directly.
    If it looks like a filename, query the graph for the matching source_file_id.
    """
    raw = ""
    for key in ("file_id", "file", "filename"):
        raw = str(extra.get(key) or "").strip()
        if raw:
            break
    if not raw:
        return None, None

    # 1) Try UUID parse
    normalized = normalize_uuid(raw)
    if normalized:
        return normalized, raw

    # 2) Treat as filename — extract basename
    basename = _basename(raw)
    if not basename:
        return None, raw

    # 3) Resolve via graph (Cypher lookup)
    if adapter is None or access_scope is None:
        return None, raw
    try:
        from core.graph_adapter.cypher import adapter_supports_cypher
        if not adapter_supports_cypher(adapter):
            return None, raw
    except Exception:
        return None, raw

    try:
        from core.graph_adapter.concurrency import adapter_locked
        # Group by source_file_id, take one metadata sample per file.
        cypher = (
            "MATCH (c:Chunk) "
            "WHERE COALESCE(c.owner_id, $global_owner) = $owner_id "
            "AND c.source_file_id IS NOT NULL "
            "WITH c.source_file_id AS file_id, COLLECT(c.metadata)[0] AS metadata "
            "RETURN file_id, metadata LIMIT 200"
        )
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, {}, access_scope=access_scope)
    except Exception:
        return None, raw

    basename_lower = basename.lower()
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            continue
        for meta_key in ("filename", "source_file_name", "file_name", "source_path"):
            candidate = str(meta.get(meta_key) or "").strip()
            if candidate and _basename(candidate).lower() == basename_lower:
                fid = str(row.get("file_id") or "").strip()
                resolved = normalize_uuid(fid)
                if resolved:
                    return resolved, raw
    return None, raw

