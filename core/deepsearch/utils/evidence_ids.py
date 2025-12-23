"""Utilities for generating stable evidence identifiers.

DeepSearch treats chunks as primary evidence. Tool-produced or synthesized evidences still need
stable, collision-resistant IDs so they can be referenced, merged, and audited reliably.
"""
import hashlib
import re
from typing import Optional


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._:-]+")


def _slug(value: str, *, default: str = "na") -> str:
    token = (value or "").strip()
    if not token:
        return default
    token = _SLUG_RE.sub("-", token)
    token = token.strip("-")
    return token or default


def _sha256_hex(*parts: str) -> str:
    joined = "\x1f".join((part or "") for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def hashed_chunk_id(*, source: str, content: str, prefix: str = "anon", length: int = 12) -> str:
    """Return a deterministic chunk_id based on source+content (used when no upstream ID exists)."""

    digest = _sha256_hex(source, content)[: max(4, int(length))]
    return f"{_slug(prefix)}-{digest}"


def derived_chunk_id(
    *,
    tool_name: str,
    plan_step: Optional[str],
    content: str,
    label: str | None = None,
    length: int = 12,
) -> str:
    """Return a deterministic ID for tool-derived evidence.

    Includes tool name + plan step + content hash to avoid collisions across repeated runs/tools.
    """

    digest = _sha256_hex(tool_name, plan_step or "", label or "", content)[: max(6, int(length))]
    tool_part = _slug(tool_name, default="tool")
    step_part = _slug(plan_step or "", default="step")
    label_part = _slug(label or "", default="ev") if label else "ev"
    return f"{tool_part}:{step_part}:{label_part}:{digest}"

