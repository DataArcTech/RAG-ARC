from typing import Any, Dict

from encapsulation.mcp.client import MCPToolClient


class DeepSearchToolManager:
    """Handles tool registration, quota policies, and logging so DeepSearch runs remain auditable."""

    def __init__(self, tool_configs, telemetry_client, *, mcp_client: MCPToolClient | None = None):
        self.tool_configs = tool_configs
        self.telemetry_client = telemetry_client
        # mcp_client: optional bridge to the shared MCP session manager used across DeepSearch services
        self.mcp_client = mcp_client

    async def invoke(self, tool_name: str, *, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool by name and record telemetry; concrete logic will reuse encapsulation layer clients."""

        raise NotImplementedError("DeepSearch Tool Manager logic will be implemented in a subsequent change")
