"""Tool hint/catalog utilities consumed by the DeepSearch planner.

Note: Infrastructure implementations (local tool registry, MCP routing, tool manager) live under
`encapsulation/deepsearch/tooling` to keep `core/` focused on algorithms and contracts.
"""
from typing import Dict, Iterable, List

from core.deepsearch.tools import builtin_tool_descriptors, llm_optional_tool_names, llm_required_tool_names

from ._hints import (
    clear_tool_hints,
    register_tool_hints,
    set_disabled_tools,
)
from .registry import DEFAULT_TOOL_HINT_REGISTRY, ToolHintRegistry
__all__ = [
    "describe_available_tools",
    "register_tool_hints",
    "set_disabled_tools",
    "clear_tool_hints",
    "get_tool_hint_revision",
]


def describe_available_tools(
    extra_hints: Iterable[Dict[str, str]] | None = None,
    *,
    registry: ToolHintRegistry | None = None,
    include_llm_tools: bool,
) -> List[Dict[str, str]]:
    """Return tool descriptors exposed to planner prompts."""

    active_registry = registry or DEFAULT_TOOL_HINT_REGISTRY
    base_hints: List[Dict[str, str]] = [desc.as_hint() for desc in builtin_tool_descriptors()]
    base_hints.extend(active_registry.get_registered_hints())
    if extra_hints:
        base_hints.extend(list(extra_hints))

    if include_llm_tools is not True and include_llm_tools is not False:
        raise ValueError("describe_available_tools requires explicit include_llm_tools boolean")
    if not include_llm_tools:
        llm_tools = llm_required_tool_names().union(llm_optional_tool_names())
        base_hints = [hint for hint in base_hints if hint.get("name") not in llm_tools]

    disabled = active_registry.get_disabled_tool_names()
    if disabled:
        base_hints = [hint for hint in base_hints if hint.get("name") not in disabled]
    return base_hints


def get_tool_hint_revision() -> int:
    """Expose hint revision timestamp for planner cache invalidation."""

    return DEFAULT_TOOL_HINT_REGISTRY.get_revision()
