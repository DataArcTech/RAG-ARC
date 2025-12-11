"""Self-contained FastMCP server that exposes DeepSearch tools."""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import FunctionTool
from mcp.types import ToolAnnotations

from config.core.deepsearch.graph_adapter_config import GraphAdapterConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from core.deepsearch.tooling import DeepSearchToolManager
from core.deepsearch.tools import ToolDescriptor, builtin_tool_descriptors
from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.scope_provider import current_scope_provider
from encapsulation.data_model.deepsearch import GraphQueryContext

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = (
    "DeepSearch tool MCP server exposing all registered tools. "
)


def _read_json_file(path: str) -> Dict[str, Any]:
    payload = Path(path).expanduser().resolve()
    data = payload.read_text(encoding="utf-8")
    return json.loads(data)


def _default_tool_names() -> Set[str]:
    return {descriptor.name for descriptor in builtin_tool_descriptors()}


def _parse_enabled_tools(raw: Optional[str]) -> Set[str]:
    if not raw:
        return _default_tool_names()
    names = {token.strip() for token in raw.split(",") if token.strip()}
    return names or _default_tool_names()


def _load_llm_from_env() -> Any:
    config = OpenAIChatConfig()
    return config.build()


def _parse_json_env(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from env, ignoring payload: %s", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_adapter_from_env():
    config_path = os.getenv("DEEPSEARCH_TOOL_MCP_ADAPTER_CONFIG")
    if config_path:
        try:
            payload = _read_json_file(config_path)
        except OSError as exc:
            raise RuntimeError(f"Failed to load adapter config from {config_path}: {exc}") from exc
    else:
        params = _parse_json_env(os.getenv("DEEPSEARCH_TOOL_MCP_ADAPTER_PARAMS"))
        payload = {
            "adapter_name": os.getenv("DEEPSEARCH_TOOL_MCP_ADAPTER_NAME")
            or os.getenv("DEEPSEARCH_DEFAULT_ADAPTER")
            or "hipporag",
            "parameters": params,
        }

    adapter_config = GraphAdapterConfig.model_validate(payload)
    return adapter_config.build()


class LoggingTelemetryClient:
    """Minimal telemetry client that emits structured logs for tool usage."""

    def log_tool_invocation(self, *, tool_name: str, payload: Dict[str, Any]) -> None:
        logger.info(
            "DeepSearch MCP tool executed locally",
            extra={
                "tool_name": tool_name,
                "plan_step": payload.get("plan_step"),
                "summary": payload.get("summary"),
                "diagnostics": payload.get("diagnostics"),
            },
        )

    def log_remote_tool(self, *, tool_name: str, log) -> None:
        logger.info(
            "DeepSearch MCP forwarded tool invocation",
            extra={
                "tool_name": tool_name,
                "server_name": log.server_name,
                "latency_ms": log.latency_ms,
                "excerpt": log.response_excerpt,
            },
        )


def _enabled_map(enabled: Set[str]) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    for descriptor in builtin_tool_descriptors():
        overrides[descriptor.name] = {"enabled": descriptor.name in enabled}
    return overrides


def build_tool_mcp_server(
    *,
    llm_connector: Any | None = None,
    enabled_tools: Optional[Sequence[str]] = None,
    instructions: Optional[str] = None,
    adapter=None,
    default_scope: GraphAccessScope | None = None,
    telemetry_client: Any | None = None,
    tool_manager_config: Optional[Dict[str, Any]] = None,
) -> "DeepSearchToolMCPServer":
    """Factory that builds the server using env defaults."""

    tool_names = set(enabled_tools or ())
    if not tool_names:
        tool_names = _parse_enabled_tools(os.getenv("DEEPSEARCH_TOOL_MCP_TOOLS"))
    llm = llm_connector or _load_llm_from_env()
    text = instructions or os.getenv("DEEPSEARCH_TOOL_MCP_INSTRUCTIONS") or DEFAULT_INSTRUCTIONS
    adapter_instance = adapter or _load_adapter_from_env()
    scope = default_scope or current_scope_provider().default_scope
    if scope is None:
        raise ValueError(
            "Graph access scope must be configured via ToolScopeConfig or DEEPSEARCH_SCOPE_* environment variables."
        )
    telemetry = telemetry_client or LoggingTelemetryClient()
    return DeepSearchToolMCPServer(
        llm_connector=llm,
        enabled_tools=tool_names,
        instructions=text,
        adapter=adapter_instance,
        default_scope=scope,
        telemetry_client=telemetry,
        tool_manager_config=tool_manager_config,
    )


class DeepSearchToolMCPServer:
    """Wraps FastMCP with DeepSearch tool registrations."""

    def __init__(
        self,
        *,
        llm_connector,
        enabled_tools: Optional[Set[str]] = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        adapter=None,
        default_scope: GraphAccessScope | None = None,
        telemetry_client=None,
        tool_manager_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if llm_connector is None:
            raise ValueError("llm_connector must be provided for tool MCP server")
        if adapter is None:
            raise ValueError("Graph adapter must be provided or configured for the MCP server")
        self.enabled_tools = self._resolve_enabled_set(enabled_tools, tool_manager_config)
        self.adapter = adapter
        self.default_scope = default_scope or current_scope_provider().default_scope
        if self.default_scope is None:
            raise ValueError("Graph access scope is required for DeepSearch tool MCP server.")
        self.adapter_name = self._resolve_adapter_name(adapter)
        self.fastmcp = FastMCP("DeepSearch Tool MCP Server", instructions=instructions)
        tool_configs = self._build_tool_manager_config(
            llm_connector=llm_connector,
            payload=tool_manager_config,
        )
        self.tool_manager = DeepSearchToolManager(
            tool_configs=tool_configs,
            telemetry_client=telemetry_client or LoggingTelemetryClient(),
        )
        self._register_tools()

    def _resolve_enabled_set(
        self,
        enabled_tools: Optional[Set[str]],
        tool_manager_config: Optional[Dict[str, Any]],
    ) -> Set[str]:
        desired = set(enabled_tools) if enabled_tools else None
        if desired is None and tool_manager_config:
            configured = tool_manager_config.get("enabled_tools")
            if isinstance(configured, dict):
                desired = {
                    name
                    for name, cfg in configured.items()
                    if not isinstance(cfg, dict) or cfg.get("enabled", True)
                }
        return self._normalize_enabled(desired)

    def _normalize_enabled(self, enabled_tools: Optional[Set[str]]) -> Set[str]:
        desired = enabled_tools or _default_tool_names()
        builtin = {descriptor.name for descriptor in builtin_tool_descriptors()}
        selected = {name for name in desired if name in builtin}
        if not selected:
            raise ValueError("At least one built-in tool must be enabled for the MCP server")
        missing = desired - builtin
        if missing:
            logger.warning("Ignoring unknown tool(s) for MCP server: %s", ", ".join(sorted(missing)))
        return selected

    def _build_tool_manager_config(
        self,
        *,
        llm_connector,
        payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        configs: Dict[str, Any] = dict(payload or {})
        configs.setdefault("enable_builtin_tools", True)
        configs["llm_connector"] = llm_connector
        configs.setdefault("audit_label", os.getenv("DEEPSEARCH_TOOL_MCP_AUDIT_LABEL"))
        if "enabled_tools" not in configs:
            configs["enabled_tools"] = _enabled_map(self.enabled_tools)
        return configs

    def _resolve_adapter_name(self, adapter) -> Optional[str]:
        metadata_getter = getattr(adapter, "metadata", None)
        if callable(metadata_getter):
            try:
                metadata = metadata_getter()
                adapter_name = getattr(metadata, "adapter_name", None)
                if adapter_name:
                    return str(adapter_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read adapter metadata: %s", exc)
        return os.getenv("DEEPSEARCH_DEFAULT_ADAPTER")

    def _register_tools(self) -> None:
        for descriptor in builtin_tool_descriptors():
            if descriptor.name not in self.enabled_tools:
                continue
            tool_callable = self._build_callable(descriptor)
            function_tool = FunctionTool(
                name=descriptor.name,
                description=descriptor.description,
                parameters=descriptor.input_schema,
                annotations=self._build_annotations(descriptor),
                tags=set(descriptor.strategy_tags),
                meta=self._build_meta(descriptor),
                fn=tool_callable,
            )
            self.fastmcp.add_tool(function_tool)
            logger.info("Registered MCP tool %s (%s)", descriptor.name, descriptor.channel)

    def _build_callable(self, descriptor: ToolDescriptor):
        async def _tool_callable(ctx: Context | None = None, **payload: Any):
            invocation_payload = self._inject_defaults(payload)
            result = await self.tool_manager.invoke(descriptor.name, payload=invocation_payload)
            return result.model_dump()

        return _tool_callable

    def _inject_defaults(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        invocation_payload = dict(payload or {})
        adapter = invocation_payload.get("adapter")
        if adapter is None:
            invocation_payload["adapter"] = self.adapter
        scope = invocation_payload.get("access_scope")
        if scope is None and self.default_scope is not None:
            invocation_payload["access_scope"] = self.default_scope
        graph_context = invocation_payload.get("graph_context")
        if graph_context is None and self.adapter_name:
            invocation_payload["graph_context"] = GraphQueryContext(
                adapter_name=self.adapter_name,
                question=invocation_payload.get("question"),
                access_scope=self.default_scope,
            )
        elif isinstance(graph_context, dict):
            normalized = dict(graph_context)
            normalized.setdefault("adapter_name", self.adapter_name)
            if self.default_scope and not normalized.get("access_scope"):
                normalized["access_scope"] = {
                    "scope_id": self.default_scope.scope_id,
                    "scope_type": self.default_scope.scope_type,
                    "labels": list(self.default_scope.labels),
                    "attributes": self.default_scope.attributes,
                }
            invocation_payload["graph_context"] = normalized
        return invocation_payload

    @staticmethod
    def _build_annotations(descriptor: ToolDescriptor) -> ToolAnnotations:
        return ToolAnnotations(
            title=descriptor.name,
            readOnlyHint=True,
            idempotentHint=True,
        )

    @staticmethod
    def _build_meta(descriptor: ToolDescriptor) -> Dict[str, Any]:
        return {
            "channel": descriptor.channel,
            "profile": descriptor.profile,
            "determinism": descriptor.determinism,
            "strategy_tags": list(descriptor.strategy_tags),
        }

    def list_registered_tools(self) -> List[ToolDescriptor]:
        """Return descriptors for tools currently exposed through FastMCP."""

        enabled = self.enabled_tools
        return [descriptor for descriptor in builtin_tool_descriptors() if descriptor.name in enabled]

    async def run_stdio_async(self) -> None:
        await self.fastmcp.run_stdio_async()

    async def run_sse_async(self, *, host: str = "127.0.0.1", port: int = 8765, path: str = "sse") -> None:
        await self.fastmcp.run_http_async(transport="sse", host=host, port=port, path=path)

    async def run_streamable_http_async(
        self, *, host: str = "127.0.0.1", port: int = 8765, path: str = "mcp"
    ) -> None:
        await self.fastmcp.run_http_async(transport="streamable-http", host=host, port=port, path=path)

    def http_app(self, *, path: str = "/mcp/tools", transport: str = "sse"):
        return self.fastmcp.http_app(path=path, transport=transport)


async def run_tool_server_stdio() -> None:
    server = build_tool_mcp_server()
    await server.run_stdio_async()


def main_stdio() -> None:
    asyncio.run(run_tool_server_stdio())
