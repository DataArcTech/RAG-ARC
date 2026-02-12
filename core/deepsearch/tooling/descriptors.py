"""Lazy accessors for DeepSearch built-in tool descriptors.

This module intentionally avoids importing ``core.deepsearch.tools`` at import time
so ``core.deepsearch.tooling`` can be imported from tool modules without circular imports.
"""

from typing import Any, Iterable, Set


def builtin_tool_descriptors() -> Iterable[Any]:
    """Return descriptors for built-in local tools (lazy import)."""

    from core.deepsearch.tools import builtin_tool_descriptors as _builtin_tool_descriptors

    return _builtin_tool_descriptors()


def llm_required_tool_names() -> Set[str]:
    """Return built-in tool names that require an LLM connector (lazy import)."""

    from core.deepsearch.tools import llm_required_tool_names as _llm_required_tool_names

    return set(_llm_required_tool_names())


def llm_optional_tool_names() -> Set[str]:
    """Return built-in tool names that optionally consume an LLM connector (lazy import)."""

    from core.deepsearch.tools import llm_optional_tool_names as _llm_optional_tool_names

    return set(_llm_optional_tool_names())


__all__ = ["builtin_tool_descriptors", "llm_required_tool_names", "llm_optional_tool_names"]
