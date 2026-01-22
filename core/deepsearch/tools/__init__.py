"""Tool definitions for DeepSearch on Graph."""
from typing import Any, Dict, Iterable, Optional

from .base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest
from .explore import (
    ExploreTool,
    GraphOpsTool,
    SearchTool,
    SearchFaissTool,
    SearchBM25Tool,
    SearchGraphChunkTool,
    WebSearchTool,
    BeamSearchTool,
    LLMChainExplorerTool,
)
from .think import ThinkTool
from .code import CodePythonTool

__all__ = [
    "GraphTool",
    "ToolDescriptor",
    "ToolResult",
    "ToolRunRequest",
    "ExploreTool",
    "GraphOpsTool",
    "SearchTool",
    "SearchFaissTool",
    "SearchBM25Tool",
    "SearchGraphChunkTool",
    "ThinkTool",
    "BeamSearchTool",
    "LLMChainExplorerTool",
    "CodePythonTool",
    "WebSearchTool",
    "build_builtin_tools",
    "builtin_tool_descriptors",
    "get_tool_descriptor",
    "llm_required_tool_names",
    "llm_optional_tool_names",
]

_BUILTIN_CLASSES = [
    GraphOpsTool,
    ExploreTool,
    SearchTool,
    SearchFaissTool,
    SearchBM25Tool,
    SearchGraphChunkTool,
    ThinkTool,
    CodePythonTool,
    WebSearchTool,
    BeamSearchTool,
    LLMChainExplorerTool,
]

_DESCRIPTOR_MAP = {cls.descriptor.name: cls.descriptor for cls in _BUILTIN_CLASSES}
_LLM_REQUIRED = {
    SearchTool,
    SearchGraphChunkTool,
    ThinkTool,
    BeamSearchTool,
    LLMChainExplorerTool,
}
_LLM_OPTIONAL = {
    ExploreTool,
}


def build_builtin_tools(llm_connector=None, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, GraphTool]:
    """Instantiate the default local graph tool set."""

    overrides = overrides or {}
    tools: Dict[str, GraphTool] = {}
    for cls in _BUILTIN_CLASSES:
        name = cls.descriptor.name
        params = dict(overrides.get(name, {}))
        if cls in _LLM_REQUIRED and llm_connector is None and "llm_connector" not in params:
            continue
        if cls in _LLM_REQUIRED or cls in _LLM_OPTIONAL:
            params.setdefault("llm_connector", llm_connector)
        tool = cls(**params)
        tools[name] = tool
    return tools


def builtin_tool_descriptors() -> Iterable[ToolDescriptor]:
    """Expose tool descriptors for think hints."""

    return [cls.descriptor for cls in _BUILTIN_CLASSES]


def get_tool_descriptor(tool_name: str) -> Optional[ToolDescriptor]:
    """Return the descriptor for built-in tools when available."""

    return _DESCRIPTOR_MAP.get(tool_name)


def llm_required_tool_names() -> set[str]:
    """Expose names of tools that require an LLM connector."""

    return {cls.descriptor.name for cls in _LLM_REQUIRED}


def llm_optional_tool_names() -> set[str]:
    """Expose names of tools that can optionally consume an LLM connector."""

    return {cls.descriptor.name for cls in _LLM_OPTIONAL}
