"""File-scope propagation policy for DeepSearch.

Why this module exists:
- Keep the "which tools are global" list centralized and configurable.
- Provide a shared helper to strip inherited file_scope from GraphQueryContext.

Non-goals:
- This does NOT enforce evidence constraints (those are handled by EvidenceBank + report writer).
"""
from typing import Any

from config.core.deepsearch.tool_defaults import (
    DEEPSEARCH_GLOBAL_ACTION_TOOL_NAMES,
    DEEPSEARCH_GLOBAL_ACTION_TOOL_PREFIXES,
)


def is_global_action_tool(tool_name: str) -> bool:
    """Return True when a tool/action should ignore inherited file_scope."""

    name = str(tool_name or "").strip()
    if not name:
        return False
    if name in set(DEEPSEARCH_GLOBAL_ACTION_TOOL_NAMES):
        return True
    for prefix in DEEPSEARCH_GLOBAL_ACTION_TOOL_PREFIXES:
        token = str(prefix or "").strip()
        if token and name.startswith(token):
            return True
    return False


def strip_file_scope_from_graph_context(graph_context: Any) -> Any:
    """Return a best-effort copy of GraphQueryContext with metadata.file_scope removed."""

    if graph_context is None:
        return None
    meta = getattr(graph_context, "metadata", None)
    if not isinstance(meta, dict) or "file_scope" not in meta:
        return graph_context
    cleaned = dict(meta)
    cleaned.pop("file_scope", None)
    try:
        return graph_context.model_copy(update={"metadata": cleaned})
    except Exception:  # noqa: BLE001
        return graph_context

