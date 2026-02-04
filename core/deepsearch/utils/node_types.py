"""Normalize semantic unit types to node categories."""
from typing import Any

from config.core.deepsearch import tool_defaults


def normalize_node_type(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    mapping = getattr(tool_defaults, "SECTION_NODE_TYPE_MAP", {}) or {}
    if token and token in mapping:
        return str(mapping[token])
    default = getattr(tool_defaults, "SECTION_NODE_TYPE_DEFAULT", "page")
    return str(default)


__all__ = ["normalize_node_type"]
