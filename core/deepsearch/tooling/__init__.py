"""Tool management utilities (MCP, telemetry, etc.)."""

import json
import os
from typing import Dict, List

from .manager import DeepSearchToolManager

__all__ = ["DeepSearchToolManager", "describe_available_tools"]

_DEFAULT_TOOL_REGISTRY: List[Dict[str, str]] = [
    {
        "name": "graph_adapter.query",
        "channel": "graph",
        "description": "HippoRAG/GraphSearch adapter entry point for retrieving subgraphs.",
    },
    {
        "name": "llm.summarize",
        "channel": "text",
        "description": "LLM summarisation helper that compacts graph evidence into prose.",
    },
    {
        "name": "web.search",
        "channel": "web",
        "description": "External web search hook (Serper/Tavily) invoked via MCP tools.",
    },
]


def describe_available_tools() -> List[Dict[str, str]]:
    """Return tool descriptors exposed to planner prompts (override via env)."""

    raw = os.getenv("DEEPSEARCH_TOOL_HINTS")
    if not raw:
        return list(_DEFAULT_TOOL_REGISTRY)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            hints = []
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    hints.append(
                        {
                            "name": str(item["name"]),
                            "channel": str(item.get("channel", "")),
                            "description": str(item.get("description", "")),
                        }
                    )
            if hints:
                return hints
    except json.JSONDecodeError:
        pass
    return list(_DEFAULT_TOOL_REGISTRY)
