"""Async MCP client with graph-aware payload helpers."""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ListToolsResult

from encapsulation.data_model.deepsearch import GraphQueryContext, ToolExecutionLog

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Lightweight representation of a tool exposed by an MCP server."""

    name: str
    description: Optional[str]
    schema: Dict[str, Any] | None


@dataclass
class MCPToolCallOutcome:
    """Wraps the raw MCP response together with an execution log."""

    result: CallToolResult
    log: ToolExecutionLog


class MCPToolClient:
    """Manage MCP tool sessions with graph-aware payload injection so downstream tools can adjust traversal or summarisation strategies for knowledge-graph centric workloads."""

    def __init__(
        self,
        *,
        server_uri: Optional[str] = None,
        transport: str = "auto",
        headers: Optional[Dict[str, str]] = None,
        stdio_command: Optional[Sequence[str]] = None,
        stdio_env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        read_timeout: float = 300.0,
        persistent_session: bool = True,
        enable_graph_context: bool = True,
        graph_context_field: str = "__graph_context__",
    ):
        self.server_uri = server_uri
        self.transport = transport
        self.headers = headers or {}
        self.stdio_command = list(stdio_command) if stdio_command else None
        self.stdio_env = stdio_env or {}
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.persistent_session = persistent_session
        self.enable_graph_context = enable_graph_context
        self.graph_context_field = graph_context_field

        self._session: Optional[ClientSession] = None
        self._client_cm = None
        self._read = None
        self._write = None
        self._lock = asyncio.Lock()
        self._resolved_transport: Optional[str] = None

    async def __aenter__(self) -> "MCPToolClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def list_tools(self) -> List[ToolDefinition]:
        """Return available tools from the configured MCP server."""
        await self._ensure_session()
        tools_response: ListToolsResult = await self._session.list_tools()
        tool_defs: List[ToolDefinition] = []
        for tool in tools_response.tools:
            tool_defs.append(
                ToolDefinition(
                    name=tool.name,
                    description=getattr(tool, "description", None),
                    schema=getattr(tool, "inputSchema", None),
                )
            )
        if not self.persistent_session:
            await self.close()
        return tool_defs

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        graph_context: Optional[GraphQueryContext] = None,
        server_name: Optional[str] = None,
    ) -> MCPToolCallOutcome:
        """Invoke a tool and optionally enrich the payload with graph context."""
        await self._ensure_session()
        payload = dict(arguments or {})
        if graph_context and self.enable_graph_context:
            payload.setdefault(
                self.graph_context_field,
                graph_context.model_dump(exclude_none=True),
            )

        start = time.perf_counter()
        result = await self._session.call_tool(tool_name, arguments=payload)
        latency_ms = int((time.perf_counter() - start) * 1000)
        outcome = MCPToolCallOutcome(
            result=result,
            log=self._build_execution_log(
                tool_name=tool_name,
                server_name=server_name,
                payload=payload,
                latency_ms=latency_ms,
                graph_context=graph_context,
                result=result,
            ),
        )
        if not self.persistent_session:
            await self.close()
        return outcome

    def is_connected(self) -> bool:
        """Return True when there is an active MCP session."""
        return self._session is not None

    async def close(self) -> None:
        """Terminate the MCP session and its transport."""
        async with self._lock:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None
            if self._client_cm:
                await self._client_cm.__aexit__(None, None, None)
                self._client_cm = None
                self._read = None
                self._write = None

    async def _ensure_session(self) -> None:
        """Create a session if none exists."""
        if self._session:
            return

        async with self._lock:
            if self._session:
                return
            self._client_cm = self._build_transport()
            self._read, self._write = await self._client_cm.__aenter__()
            self._session = ClientSession(self._read, self._write, sampling_callback=None)
            await self._session.__aenter__()
            await self._session.initialize()
            logger.info("Initialized MCP session via %s transport", self._resolved_transport)

    def _build_transport(self):
        """Return the async context manager for the current transport."""
        transport = self._resolve_transport()
        self._resolved_transport = transport
        if transport == "sse":
            if not self.server_uri:
                raise ValueError("server_uri is required for SSE transport")
            return sse_client(
                self.server_uri,
                headers=self.headers or None,
                timeout=self.timeout,
                sse_read_timeout=self.read_timeout,
            )
        if transport == "stdio":
            params = self._build_stdio_params()
            return stdio_client(params)
        raise ValueError(f"Unsupported transport: {transport}")

    def _build_stdio_params(self) -> StdioServerParameters:
        """Construct stdio server parameters."""
        if not self.stdio_command:
            raise ValueError("stdio_command must be provided when using stdio transport")
        command, *args = self.stdio_command
        return StdioServerParameters(
            command=command,
            args=args,
            env=self.stdio_env,
        )

    def _resolve_transport(self) -> str:
        """Determine which transport to use."""
        if self.transport != "auto":
            return self.transport
        if self.server_uri and self.server_uri.startswith(("http://", "https://")):
            return "sse"
        if self.stdio_command:
            return "stdio"
        raise ValueError("Cannot auto-detect transport without server_uri or stdio_command")

    def _build_execution_log(
        self,
        *,
        tool_name: str,
        server_name: Optional[str],
        payload: Dict[str, Any],
        latency_ms: int,
        graph_context: Optional[GraphQueryContext],
        result: CallToolResult,
    ) -> ToolExecutionLog:
        """Create a ToolExecutionLog snapshot for telemetry."""
        excerpt = self._extract_text_excerpt(result)
        return ToolExecutionLog(
            tool_name=tool_name,
            server_name=server_name,
            arguments_snapshot=payload,
            response_excerpt=excerpt,
            latency_ms=latency_ms,
            graph_context=graph_context,
            extra={"transport": getattr(self, "_resolved_transport", self.transport)},
        )

    @staticmethod
    def _extract_text_excerpt(result: CallToolResult) -> Optional[str]:
        """Extract a short textual snippet from the MCP response."""
        if not result.content:
            return None
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                return text[:512]
        return None
