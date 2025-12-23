"""Backward-compatible module-level wrappers for tool hint registry."""

from typing import Dict, Iterable, List, Set

from .registry import DEFAULT_TOOL_HINT_REGISTRY


def register_tool_hints(hints: Iterable[Dict[str, str]]) -> None:
    """Store extra tool hints so planner layer can see new tools(e.g. MCP-Only)"""
    DEFAULT_TOOL_HINT_REGISTRY.register_tool_hints(hints)


def set_disabled_tools(names: Iterable[str]) -> None:
    """Replace the disabled tool set so planner hints can filter them out."""
    DEFAULT_TOOL_HINT_REGISTRY.set_disabled_tools(names)


def get_registered_hints() -> List[Dict[str, str]]:
    """Return a copy of the registered hints."""
    return DEFAULT_TOOL_HINT_REGISTRY.get_registered_hints()


def clear_tool_hints() -> None:
    """Reset registered hints (used mainly in tests)."""
    DEFAULT_TOOL_HINT_REGISTRY.clear()


def get_disabled_tool_names() -> Set[str]:
    """Return tool names that should be hidden from planner prompts."""
    return DEFAULT_TOOL_HINT_REGISTRY.get_disabled_tool_names()


def get_hint_revision() -> int:
    """Return the current revision counter for tool hints."""
    return DEFAULT_TOOL_HINT_REGISTRY.get_revision()
