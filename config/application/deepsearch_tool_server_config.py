"""Configuration for DeepSearch Tool MCP server."""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from application.deepsearch.tool_mcp_server import build_tool_mcp_server, DeepSearchToolMCPServer
from config.application.deepsearch_config import ToolManagerConfig
from config.core.deepsearch.graph_adapter_config import GraphAdapterConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from core.graph_adapter.base import GraphAccessScope
from framework.config import AbstractConfig


class ToolScopeConfig(BaseModel):
    """Declarative GraphAccessScope configuration."""

    scope_id: str = Field(..., description="Tenant/user identifier passed to graph adapter")
    scope_type: str = Field("owner", description="Scope type label, defaults to owner")
    labels: List[str] = Field(default_factory=list, description="Additional labels forwarded to the adapter")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata for adapter policies")

    def to_scope(self) -> GraphAccessScope:
        """Return GraphAccessScope instance."""

        return GraphAccessScope(
            scope_id=str(self.scope_id),
            scope_type=self.scope_type,
            labels=tuple(self.labels),
            attributes=self.attributes or None,
        )


class DeepSearchToolServerConfig(AbstractConfig):
    """Top-level config that wires LLM, adapter, and tool manager for the MCP server."""

    type: Literal["deepsearch_tool_mcp_server"] = "deepsearch_tool_mcp_server"
    instructions: Optional[str] = Field(
        None,
        description="Overrides default MCP instructions shown to upstream clients",
    )
    enabled_tools: Optional[List[str]] = Field(
        default=None,
        description="Optional whitelist; empty means expose all builtin DeepSearch tools",
    )
    llm_config: OpenAIChatConfig
    graph_adapter: GraphAdapterConfig
    tool_manager: ToolManagerConfig = Field(default_factory=ToolManagerConfig)
    scope: Optional[ToolScopeConfig] = Field(
        default=None,
        description="Optional default GraphAccessScope propagated to tools",
    )

    def build(self) -> DeepSearchToolMCPServer:
        """Instantiate DeepSearchToolMCPServer using nested configs."""

        llm = self.llm_config.build()
        adapter = self.graph_adapter.build()
        scope = self.scope.to_scope() if self.scope else None
        tool_manager_payload = self.tool_manager.model_dump()
        return build_tool_mcp_server(
            llm_connector=llm,
            enabled_tools=self.enabled_tools,
            instructions=self.instructions,
            adapter=adapter,
            default_scope=scope,
            tool_manager_config=tool_manager_payload,
        )


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_tool_server_config(path: str | Path) -> DeepSearchToolServerConfig:
    """Helper used by CLI/servers to parse config JSON."""

    payload_path = Path(path).expanduser().resolve()
    raw_text = payload_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    substituted = _substitute_env(data)
    return DeepSearchToolServerConfig.model_validate(substituted)


def _substitute_env(obj: Any):
    """Apply ${VAR} substitutions similar to Register."""

    if isinstance(obj, dict):
        return {key: _substitute_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env(item) for item in obj]
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(_replace_env, obj)
    return obj


def _replace_env(match: re.Match) -> str:
    var = match.group(1)
    value = os.getenv(var)
    if value is None:
        return match.group(0)
    return value
