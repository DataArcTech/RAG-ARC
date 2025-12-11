"""Tool management utilities (MCP, telemetry, etc.)."""
import json
import os
from typing import Dict, Iterable, List

from core.deepsearch.tools import builtin_tool_descriptors

from ._hints import (
    clear_tool_hints,
    get_disabled_tool_names,
    get_hint_revision,
    get_registered_hints,
    register_tool_hints,
    set_disabled_tools,
)
from .manager import DeepSearchToolManager, LocalToolRegistry, MCPToolRouter

__all__ = [
    "DeepSearchToolManager",
    "LocalToolRegistry",
    "MCPToolRouter",
    "describe_available_tools",
    "register_tool_hints",
    "set_disabled_tools",
    "clear_tool_hints",
    "get_tool_hint_revision",
]


def describe_available_tools(extra_hints: Iterable[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
    """Return tool descriptors exposed to planner prompts (override via env/config)."""

    base_hints: List[Dict[str, str]] = [desc.as_hint() for desc in builtin_tool_descriptors()]
    base_hints.extend(get_registered_hints())
    if extra_hints:
        base_hints.extend(list(extra_hints))

    raw = os.getenv("DEEPSEARCH_TOOL_HINTS")
    env_hints = _parse_env_hints(raw) if raw else None
    if env_hints is not None:
        base_hints = env_hints + base_hints

    disabled = get_disabled_tool_names()
    if disabled:
        base_hints = [hint for hint in base_hints if hint.get("name") not in disabled]
    return base_hints


def _parse_env_hints(raw: str | None) -> List[Dict[str, str]] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    env_hints: List[Dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict) and "name" in item:
            env_hints.append(
                {
                    "name": str(item["name"]),
                    "channel": str(item.get("channel", "")),
                    "description": str(item.get("description", "")),
                    "profile": str(item.get("profile", "")),
                    "determinism": str(item.get("determinism", "")),
                    "namespace": str(item.get("namespace", "")),
                    "speed": str(item.get("speed", "")),
                    "cost": str(item.get("cost", "")),
                    "strategy_tags": list(item.get("strategy_tags", [])),
                }
            )
    return env_hints


def get_tool_hint_revision() -> int:
    """Expose hint revision timestamp for planner cache invalidation."""

    return get_hint_revision()
