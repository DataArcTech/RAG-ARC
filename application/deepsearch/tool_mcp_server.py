"""Self-contained FastMCP server that exposes DeepSearch tools."""
import asyncio
import json
import logging
import time
import argparse
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import FunctionTool
from mcp.types import ToolAnnotations

from config.application import deepsearch_tool_server_defaults as tool_server_defaults
from config.application.deepsearch_config import ToolManagerConfig
from encapsulation.deepsearch.telemetry import LoggingTelemetryClient
from encapsulation.deepsearch.tooling import DeepSearchToolManager
from core.deepsearch.tools import GraphTool, ToolDescriptor, builtin_tool_descriptors
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.deepsearch import GraphQueryContext

logger = logging.getLogger(__name__)

def _enabled_map(enabled: Set[str]) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    for descriptor in builtin_tool_descriptors():
        overrides[descriptor.name] = {"enabled": descriptor.name in enabled}
    return overrides


def build_tool_mcp_server(
    *,
    llm_connector: Any,
    enabled_tools: Optional[Sequence[str]] = None,
    instructions: Optional[str] = None,
    adapter=None,
    default_scope: GraphAccessScope | None = None,
    telemetry_client: Any | None = None,
    tool_manager_config: Optional[Dict[str, Any]] = None,
    local_tools: Optional[Dict[str, GraphTool]] = None,
    scope_override_policy: Optional[str] = None,
    scope_override_token: Optional[str] = None,
) -> "DeepSearchToolMCPServer":
    """Factory that builds the server from explicit inputs (no env fallback)."""

    tool_names = set(enabled_tools or ())
    if not tool_names:
        raise ValueError("enabled_tools is required (no implicit defaults).")
    text = (instructions or "").strip()
    if not text:
        raise ValueError("instructions is required (no implicit defaults).")
    if adapter is None:
        raise ValueError("adapter is required (no implicit defaults).")
    scope = default_scope
    if scope is None:
        raise ValueError("default_scope is required (no implicit defaults).")
    telemetry = telemetry_client or LoggingTelemetryClient()
    return DeepSearchToolMCPServer(
        llm_connector=llm_connector,
        enabled_tools=tool_names,
        instructions=text,
        adapter=adapter,
        default_scope=scope,
        telemetry_client=telemetry,
        tool_manager_config=tool_manager_config,
        local_tools=local_tools,
        scope_override_policy=scope_override_policy,
        scope_override_token=scope_override_token,
    )


class DeepSearchToolMCPServer:
    """Wraps FastMCP with DeepSearch tool registrations."""

    def __init__(
        self,
        *,
        llm_connector,
        enabled_tools: Optional[Set[str]] = None,
        instructions: str,
        adapter=None,
        default_scope: GraphAccessScope | None = None,
        telemetry_client=None,
        tool_manager_config: Optional[Dict[str, Any]] = None,
        local_tools: Optional[Dict[str, GraphTool]] = None,
        scope_override_policy: Optional[str] = None,
        scope_override_token: Optional[str] = None,
    ) -> None:
        if llm_connector is None:
            raise ValueError("llm_connector must be provided for tool MCP server")
        if adapter is None:
            raise ValueError("Graph adapter must be provided or configured for the MCP server")
        self.enabled_tools = self._validate_enabled_tools(enabled_tools)
        self.adapter = adapter
        if default_scope is None:
            raise ValueError("Graph access scope is required for DeepSearch tool MCP server.")
        self.default_scope = default_scope
        self.adapter_name = self._resolve_adapter_name(adapter)
        self._scope_override_policy = self._normalize_scope_override_policy(
            scope_override_policy if scope_override_policy is not None else tool_server_defaults.DEFAULT_SCOPE_OVERRIDE_POLICY
        )
        self._scope_override_token = scope_override_token
        self.fastmcp = FastMCP("DeepSearch Tool MCP Server", instructions=instructions)
        tool_configs = self._build_tool_manager_config(
            llm_connector=llm_connector,
            payload=tool_manager_config,
        )
        self.tool_manager = DeepSearchToolManager(
            tool_configs=tool_configs,
            telemetry_client=telemetry_client or LoggingTelemetryClient(),
            local_tools=local_tools,
        )
        self._register_tools()

    @staticmethod
    def _normalize_scope_override_policy(raw: Optional[str]) -> str:
        policy = (raw or "").strip().lower()
        if policy in {"ignore", "allow_trusted", "allow_all"}:
            return policy
        raise ValueError(f"Unknown scope override policy: {raw}")

    def _validate_enabled_tools(self, enabled_tools: Optional[Set[str]]) -> Set[str]:
        desired = set(enabled_tools or ())
        builtin = {descriptor.name for descriptor in builtin_tool_descriptors()}
        if not desired:
            raise ValueError("enabled_tools is required for the MCP server (no implicit defaults).")
        missing = {name for name in desired if name not in builtin}
        if missing:
            raise ValueError(f"Unknown tool(s) requested for MCP server: {', '.join(sorted(missing))}")
        selected = set(desired)
        if not selected:
            raise ValueError("At least one built-in tool must be enabled for the MCP server")
        return selected

    def _build_tool_manager_config(
        self,
        *,
        llm_connector,
        payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        configs: Dict[str, Any] = dict(payload or {})
        configs["llm_connector"] = llm_connector
        configs.setdefault("enabled_tools", _enabled_map(self.enabled_tools))
        cfg = ToolManagerConfig.model_validate(configs)
        if not (cfg.artifact_dir or ""):
            raise ValueError("tool_manager_config.artifact_dir is required (no implicit defaults).")
        return cfg.model_dump()

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
        return None

    def _register_tools(self) -> None:
        for descriptor in builtin_tool_descriptors():
            if descriptor.name not in self.enabled_tools:
                continue
            tool_callable = self._build_callable(descriptor)
            mcp_tool_name = descriptor.namespace or descriptor.name
            function_tool = FunctionTool(
                name=mcp_tool_name,
                description=descriptor.description,
                parameters=descriptor.input_schema,
                annotations=self._build_annotations(descriptor),
                tags=set(descriptor.strategy_tags),
                meta=self._build_meta(descriptor),
                fn=tool_callable,
            )
            self.fastmcp.add_tool(function_tool)
            logger.info("Registered MCP tool %s (logical=%s)", mcp_tool_name, descriptor.name)

    def _build_callable(self, descriptor: ToolDescriptor):
        async def _tool_callable(ctx: Context | None = None, **payload: Any):
            start = time.perf_counter()
            invocation_payload, audit = self._inject_defaults(payload)
            result = await self.tool_manager.invoke(descriptor.name, payload=invocation_payload)
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._log_invocation(
                descriptor=descriptor,
                graph_context=invocation_payload.get("graph_context"),
                latency_ms=latency_ms,
                evidence_count=len(result.evidences or []),
                audit=audit,
            )
            return result.model_dump()

        return _tool_callable

    def _inject_defaults(self, payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        invocation_payload = dict(payload or {})
        adapter = invocation_payload.get("adapter")
        if adapter is None:
            invocation_payload["adapter"] = self.adapter
        audit = self._apply_scope_override_policy(invocation_payload)
        graph_context = invocation_payload.get("graph_context")
        if graph_context is None and self.adapter_name:
            invocation_payload["graph_context"] = GraphQueryContext(
                adapter_name=self.adapter_name,
                question=invocation_payload.get("question"),
                access_scope=self.default_scope,
            )
        elif isinstance(graph_context, GraphQueryContext):
            updates: Dict[str, Any] = {}
            if self.adapter_name and not graph_context.adapter_name:
                updates["adapter_name"] = self.adapter_name
            if self.default_scope and (audit.get("enforced_scope") or graph_context.access_scope is None):
                updates["access_scope"] = self.default_scope
            if updates:
                invocation_payload["graph_context"] = graph_context.model_copy(update=updates)
        elif isinstance(graph_context, dict):
            normalized = dict(graph_context)
            normalized.setdefault("adapter_name", self.adapter_name)
            if self.default_scope and (audit.get("enforced_scope") or not normalized.get("access_scope")):
                normalized["access_scope"] = self._scope_payload(self.default_scope)
            invocation_payload["graph_context"] = normalized
        return invocation_payload, audit

    def _apply_scope_override_policy(self, invocation_payload: Dict[str, Any]) -> Dict[str, Any]:
        requested_scope = invocation_payload.get("access_scope")
        requested_scope_present = requested_scope is not None
        token = self._extract_scope_override_token(invocation_payload)
        allow_override = False
        if self._scope_override_policy == "allow_all":
            allow_override = True
        elif self._scope_override_policy == "allow_trusted":
            allow_override = bool(self._scope_override_token and token and token == self._scope_override_token)

        enforced = False
        if self.default_scope is not None and (self._scope_override_policy == "ignore" or not allow_override):
            invocation_payload["access_scope"] = self.default_scope
            enforced = True
        elif requested_scope is None and self.default_scope is not None:
            invocation_payload["access_scope"] = self.default_scope

        return {
            "scope_override_policy": self._scope_override_policy,
            "scope_override_allowed": allow_override,
            "scope_override_requested": requested_scope_present,
            "enforced_scope": enforced,
        }

    def _extract_scope_override_token(self, invocation_payload: Dict[str, Any]) -> Optional[str]:
        token = invocation_payload.pop("scope_override_token", None)
        extra = invocation_payload.get("extra")
        if isinstance(extra, dict):
            token = token or extra.pop("scope_override_token", None)
        token = token or None
        if token is None:
            return None
        return str(token)

    @staticmethod
    def _scope_payload(scope: GraphAccessScope) -> Dict[str, Any]:
        return {
            "scope_id": scope.scope_id,
            "scope_type": scope.scope_type,
            "labels": list(scope.labels),
            "attributes": scope.attributes,
        }

    def _log_invocation(
        self,
        *,
        descriptor: ToolDescriptor,
        graph_context: Any,
        latency_ms: int,
        evidence_count: int,
        audit: Dict[str, Any],
    ) -> None:
        run_id = None
        if isinstance(graph_context, dict):
            run_id = ((graph_context.get("metadata") or {}).get("run_id")) or None
        elif hasattr(graph_context, "metadata"):
            metadata = getattr(graph_context, "metadata") or {}
            if isinstance(metadata, dict):
                run_id = metadata.get("run_id")
        logger.info(
            "deepsearch.tool_mcp",
            extra={
                "event": "tool_mcp",
                "run_id": run_id,
                "tool_name": descriptor.name,
                "tool_namespace": descriptor.namespace,
                "latency_ms": latency_ms,
                "evidence_count": evidence_count,
                "scope_override_allowed": audit.get("scope_override_allowed"),
                "scope_override_policy": audit.get("scope_override_policy"),
            },
        )

    @staticmethod
    def _build_annotations(descriptor: ToolDescriptor) -> ToolAnnotations:
        return ToolAnnotations(
            title=descriptor.namespace or descriptor.name,
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

    async def run_sse_async(
        self,
        *,
        host: str = tool_server_defaults.DEFAULT_HOST,
        port: int = tool_server_defaults.DEFAULT_PORT,
        path: str = tool_server_defaults.DEFAULT_SSE_PATH,
    ) -> None:
        normalized = (path or "").strip()
        if normalized and not normalized.startswith("/"):
            normalized = "/" + normalized
        await self.fastmcp.run_http_async(transport="sse", host=host, port=port, path=normalized or None)

    async def run_streamable_http_async(
        self,
        *,
        host: str = tool_server_defaults.DEFAULT_HOST,
        port: int = tool_server_defaults.DEFAULT_PORT,
        path: str = tool_server_defaults.DEFAULT_MCP_PATH,
    ) -> None:
        normalized = (path or "").strip()
        if normalized and not normalized.startswith("/"):
            normalized = "/" + normalized
        await self.fastmcp.run_http_async(transport="streamable-http", host=host, port=port, path=normalized or None)

    def http_app(
        self,
        *,
        path: str = tool_server_defaults.DEFAULT_HTTP_APP_PATH,
        transport: str = tool_server_defaults.DEFAULT_HTTP_APP_TRANSPORT,
    ):
        return self.fastmcp.http_app(path=path, transport=transport)


async def run_tool_server_stdio() -> None:
    raise RuntimeError("run_tool_server_stdio requires an explicit --config path (no implicit defaults).")


def main_stdio() -> None:
    parser = argparse.ArgumentParser(description="Run DeepSearch tool MCP server (stdio).")
    parser.add_argument("--config", required=True, help="Path to tool server JSON config")
    args = parser.parse_args()
    from config.application.deepsearch_tool_server_config import load_tool_server_config

    config = load_tool_server_config(args.config)
    server = config.build()
    asyncio.run(server.run_stdio_async())
