"""Tool definitions for DeepSearch on Graph."""
from typing import Any, Dict, Iterable, Optional

from .base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest
from .explore import (
    LocateTool,
    TocTreeTool,
    ReadPagesTool,
    WebSearchTool,
)
from .think import ThinkTool
from .code import CodePythonTool

__all__ = [
    "GraphTool",
    "ToolDescriptor",
    "ToolResult",
    "ToolRunRequest",
    "LocateTool",
    "TocTreeTool",
    "ReadPagesTool",
    "ThinkTool",
    "CodePythonTool",
    "WebSearchTool",
    "build_builtin_tools",
    "builtin_tool_descriptors",
    "get_tool_descriptor",
    "llm_required_tool_names",
    "llm_optional_tool_names",
]

_BUILTIN_CLASSES = [
    LocateTool,
    TocTreeTool,
    ReadPagesTool,
    ThinkTool,
    CodePythonTool,
    WebSearchTool,
]

_DESCRIPTOR_MAP = {cls.descriptor.name: cls.descriptor for cls in _BUILTIN_CLASSES}
_LLM_REQUIRED = {
    ThinkTool,
}
_LLM_OPTIONAL = {
    LocateTool,
}


def build_builtin_tools(llm_connector=None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, GraphTool]:
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
    return [cls.descriptor for cls in _BUILTIN_CLASSES]


def get_tool_descriptor(tool_name: str) -> Optional[ToolDescriptor]:
    return _DESCRIPTOR_MAP.get(tool_name)


def llm_required_tool_names() -> set[str]:
    return {cls.descriptor.name for cls in _LLM_REQUIRED}


def llm_optional_tool_names() -> set[str]:
    return {cls.descriptor.name for cls in _LLM_OPTIONAL}
