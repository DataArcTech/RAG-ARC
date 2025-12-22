"""Tool hint/catalog utilities consumed by the DeepSearch planner.

Note: Infrastructure implementations (local tool registry, MCP routing, tool manager) live under
`encapsulation/deepsearch/tooling` to keep `core/` focused on algorithms and contracts.
"""
import json
import os
from typing import Dict, Iterable, List

from core.deepsearch.tools import builtin_tool_descriptors, llm_required_tool_names

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
) -> List[Dict[str, str]]:
    """Return tool descriptors exposed to planner prompts (override via env/config)."""

    active_registry = registry or DEFAULT_TOOL_HINT_REGISTRY
    base_hints: List[Dict[str, str]] = [desc.as_hint() for desc in builtin_tool_descriptors()]
    base_hints.extend(active_registry.get_registered_hints())
    if extra_hints:
        base_hints.extend(list(extra_hints))

    raw = os.getenv("DEEPSEARCH_TOOL_HINTS")
    env_hints = _parse_env_hints(raw) if raw else None
    if env_hints is not None:
        base_hints = env_hints + base_hints

    if not _llm_tools_enabled():
        llm_only = llm_required_tool_names()
        base_hints = [hint for hint in base_hints if hint.get("name") not in llm_only]

    disabled = active_registry.get_disabled_tool_names()
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

    return DEFAULT_TOOL_HINT_REGISTRY.get_revision()


def _llm_tools_enabled() -> bool:
    """Determine whether LLM-dependent tools should be advertised."""

    disable_env = _parse_env_flag("DEEPSEARCH_DISABLE_LLM_TOOLS")
    if disable_env is True:
        return False

    enable_env = _parse_env_flag("DEEPSEARCH_ENABLE_LLM_TOOLS")
    if enable_env is not None:
        return enable_env

    # Heuristic: expose LLM tools only when a known API key is available.
    llm_env_vars = (
        "CHAT_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_API_KEY",
        "AWS_BEDROCK_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "TOGETHER_API_KEY",
        "QWEN_API_KEY",
        "DEEPSEEK_API_KEY",
    )
    return any(os.getenv(var) for var in llm_env_vars)


def _parse_env_flag(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
