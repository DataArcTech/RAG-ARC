"""Configuration schema for the MCPToolClient."""
import json
import os
import shlex
from typing import Dict, List, Literal, Optional

from pydantic import Field

from encapsulation.mcp.client import MCPToolClient
from framework.config import AbstractConfig


class MCPClientConfig(AbstractConfig):
    """Declare MCP client parameters with graph-aware defaults."""

    type: Literal["mcp_client"] = "mcp_client"
    enabled: bool = Field(True, description="Master switch for MCP client creation")
    server_uri: Optional[str] = Field(None, description="HTTP(S) endpoint for SSE transport")
    transport: Literal["auto", "sse", "stdio"] = Field(
        "auto", description="Transport selection strategy for the MCP client"
    )
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers added to SSE requests")
    stdio_command: Optional[List[str]] = Field(
        None, description="Command executed when using stdio transport (first item = executable)"
    )
    stdio_env: Dict[str, str] = Field(default_factory=dict, description="Environment variables for stdio transport")
    api_key_env: Optional[str] = Field(None, description="Environment variable name holding the MCP API key")
    timeout: float = Field(30.0, description="Connection timeout for opening MCP sessions")
    read_timeout: float = Field(300.0, description="Read timeout for SSE transports")
    persistent_session: bool = Field(True, description="Keep the MCP session alive between tool calls")
    enable_graph_context: bool = Field(True, description="Inject GraphQueryContext into payloads when provided")
    graph_context_field: str = Field(
        "__graph_context__", description="Key used when embedding GraphQueryContext inside tool arguments"
    )

    def build(self) -> MCPToolClient | None:
        env = os.getenv

        def _coerce_bool(value: Optional[str], default: bool) -> bool:
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        def _parse_json(value: Optional[str]) -> Dict[str, str]:
            if not value:
                return {}
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass
            return {}

        enabled_flag = self.enabled
        env_enabled = env("DEEPSEARCH_MCP_ENABLED")
        if env_enabled is not None:
            enabled_flag = _coerce_bool(env_enabled, self.enabled)
        if not enabled_flag:
            return None

        server_uri = self.server_uri or env("DEEPSEARCH_MCP_SERVER_URI")
        transport = env("DEEPSEARCH_MCP_TRANSPORT") or self.transport
        stdio_cmd_env = env("DEEPSEARCH_MCP_STDIO_COMMAND")
        stdio_command = (
            self.stdio_command
            if self.stdio_command
            else shlex.split(stdio_cmd_env) if stdio_cmd_env else None
        )
        stdio_env = self.stdio_env or _parse_json(env("DEEPSEARCH_MCP_STDIO_ENV"))

        api_key = env(self.api_key_env) if self.api_key_env else env("DEEPSEARCH_MCP_API_KEY")

        headers = dict(self.headers)
        headers.update(_parse_json(env("DEEPSEARCH_MCP_HEADERS")))
        if api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = float(env("DEEPSEARCH_MCP_TIMEOUT") or self.timeout)
        read_timeout = float(env("DEEPSEARCH_MCP_READ_TIMEOUT") or self.read_timeout)
        persistent_session = _coerce_bool(
            env("DEEPSEARCH_MCP_PERSISTENT_SESSION"), self.persistent_session
        )
        enable_graph_context = _coerce_bool(
            env("DEEPSEARCH_MCP_ENABLE_GRAPH_CONTEXT"), self.enable_graph_context
        )
        graph_context_field = env("DEEPSEARCH_MCP_GRAPH_CONTEXT_FIELD") or self.graph_context_field

        if not server_uri and not stdio_command:
            # No transport configured: treat as disabled to unblock graph-only mode.
            return None

        return MCPToolClient(
            server_uri=server_uri,
            transport=transport,
            headers=headers,
            stdio_command=stdio_command,
            stdio_env=stdio_env,
            timeout=timeout,
            read_timeout=read_timeout,
            persistent_session=persistent_session,
            enable_graph_context=enable_graph_context,
            graph_context_field=graph_context_field,
        )
