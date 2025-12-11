"""Utility helpers for sharing DeepSearch tool descriptors with planners."""
from typing import Dict, Iterable, List, Set

_ADDITIONAL_HINTS: List[Dict[str, str]] = []
_DISABLED_TOOLS: Set[str] = set()
_HINT_REVISION: int = 0


def _bump_revision() -> None:
    global _HINT_REVISION
    _HINT_REVISION += 1


def register_tool_hints(hints: Iterable[Dict[str, str]]) -> None:
    """Store extra tool hints so planner layer can see new tools(e.g. MCP-Only)"""

    for hint in hints:
        if not isinstance(hint, dict) or "name" not in hint:
            continue
        name = str(hint["name"])
        normalized = {
            "name": name,
            "channel": str(hint.get("channel", "")),
            "description": str(hint.get("description", "")),
            "profile": str(hint.get("profile", "")),
            "determinism": str(hint.get("determinism", "")),
            "namespace": str(hint.get("namespace", "")),
            "speed": str(hint.get("speed", "")),
            "cost": str(hint.get("cost", "")),
            "strategy_tags": list(hint.get("strategy_tags", [])),
        }
        _replace_or_append(normalized)
    if hints:
        _bump_revision()


def set_disabled_tools(names: Iterable[str]) -> None:
    """Replace the disabled tool set so planner hints can filter them out."""

    normalized = {str(name) for name in names if str(name).strip()}
    global _DISABLED_TOOLS
    if normalized == _DISABLED_TOOLS:
        return
    _DISABLED_TOOLS = normalized
    _bump_revision()


def get_registered_hints() -> List[Dict[str, str]]:
    """Return a copy of the registered hints."""

    return list(_ADDITIONAL_HINTS)


def clear_tool_hints() -> None:
    """Reset registered hints (used mainly in tests)."""

    _ADDITIONAL_HINTS.clear()
    _DISABLED_TOOLS.clear()
    _bump_revision()


def get_disabled_tool_names() -> Set[str]:
    """Return tool names that should be hidden from planner prompts."""

    return set(_DISABLED_TOOLS)


def get_hint_revision() -> int:
    """Return the current revision counter for tool hints."""

    return _HINT_REVISION


def _replace_or_append(hint: Dict[str, str]) -> None:
    for idx, existing in enumerate(_ADDITIONAL_HINTS):
        if existing.get("name") == hint["name"]:
            _ADDITIONAL_HINTS[idx] = hint
            return
    _ADDITIONAL_HINTS.append(hint)
