"""Configuration for DeepSearch Tool MCP server."""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from application.deepsearch.tool_mcp_server import build_tool_mcp_server, DeepSearchToolMCPServer
from config.application.deepsearch_config import DeepSearchServiceConfig, ToolManagerConfig
from config.core.deepsearch.graph_adapter_config import GraphAdapterConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from core.deepsearch.tools import builtin_tool_descriptors
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
    instructions: str = Field(..., description="MCP instructions shown to upstream clients")
    enabled_tools: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional whitelist of DeepSearch tool names to expose. "
            "When omitted, defaults to a minimal set derived from `DEEPSEARCH_SERVICE_CONFIG_PATH`."
        ),
    )
    inherit_from_deepsearch_service: bool = Field(
        True,
        description=(
            "When true, missing `llm_config`/`graph_adapter`/`tool_manager.enabled_tools` values are sourced "
            "from the DeepSearch service config (DEEPSEARCH_SERVICE_CONFIG_PATH)."
        ),
    )
    llm_config: Optional[OpenAIChatConfig] = Field(
        default=None,
        description="Optional: explicit LLM config; when omitted and inherit_from_deepsearch_service=true, reuses DeepSearch service config.",
    )
    graph_adapter: Optional[GraphAdapterConfig] = Field(
        default=None,
        description="Optional: explicit graph adapter config; when omitted and inherit_from_deepsearch_service=true, reuses DeepSearch service config.",
    )
    tool_manager: Optional[ToolManagerConfig] = Field(
        default=None,
        description=(
            "Optional: tool manager overrides. When omitted and inherit_from_deepsearch_service=true, "
            "the MCP server uses the DeepSearch service tool_manager config."
        ),
    )
    scope: Optional[ToolScopeConfig] = Field(
        default=None,
        description="Optional default GraphAccessScope propagated to tools",
    )

    def build(self) -> DeepSearchToolMCPServer:
        """Instantiate DeepSearchToolMCPServer using nested configs."""

        service_config: DeepSearchServiceConfig | None = None
        if self.inherit_from_deepsearch_service and (
            self.llm_config is None
            or self.graph_adapter is None
            or self.enabled_tools is None
            or self.tool_manager is None
        ):
            service_config = _load_deepsearch_service_config(_resolve_deepsearch_service_config_path())

        llm_cfg = self.llm_config or (service_config.llm if service_config else None)
        if llm_cfg is None:
            raise ValueError("llm_config is required (set explicitly or enable inherit_from_deepsearch_service).")
        llm = llm_cfg.build()

        adapter_cfg = self.graph_adapter or (service_config.graph_adapter if service_config else None)
        if adapter_cfg is None:
            raise ValueError("graph_adapter is required (set explicitly or enable inherit_from_deepsearch_service).")
        adapter = adapter_cfg.build()

        scope = self.scope.to_scope() if self.scope else None
        tool_manager_payload = _merge_tool_manager_payload(
            service_tool_manager=service_config.tool_manager if service_config else None,
            overrides=self.tool_manager,
        )
        enabled_tools = _resolve_enabled_tools(
            explicit=self.enabled_tools,
            service_config=service_config,
        )
        return build_tool_mcp_server(
            llm_connector=llm,
            enabled_tools=enabled_tools,
            instructions=self.instructions,
            adapter=adapter,
            default_scope=scope,
            tool_manager_config=tool_manager_payload,
        )


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
_UNRESOLVED_ENV_PLACEHOLDER = object()

try:  # Keep substitution rules consistent with framework.Register
    from config.env_placeholder_policy import ENV_DEFAULTS as _ENV_DEFAULTS
    from config.env_placeholder_policy import SILENT_MISSING_ENV_VARS as _SILENT_MISSING_ENV_VARS
except Exception:  # pragma: no cover - defensive for minimal runtimes
    _ENV_DEFAULTS = {}
    _SILENT_MISSING_ENV_VARS = set()


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
        resolved: Dict[str, Any] = {}
        for key, value in obj.items():
            substituted = _substitute_env(value)
            if substituted is _UNRESOLVED_ENV_PLACEHOLDER:
                continue
            resolved[key] = substituted
        return resolved
    if isinstance(obj, list):
        items = [_substitute_env(item) for item in obj]
        return [None if item is _UNRESOLVED_ENV_PLACEHOLDER else item for item in items]
    if isinstance(obj, str):
        whole = re.fullmatch(r"\$\{([^}]+)\}", obj.strip())
        if whole:
            var_name = whole.group(1)
            env_value = os.getenv(var_name)
            if not env_value:
                default_value = _ENV_DEFAULTS.get(var_name)
                if default_value is not None:
                    return default_value
                if var_name in _SILENT_MISSING_ENV_VARS:
                    return _UNRESOLVED_ENV_PLACEHOLDER
                return _UNRESOLVED_ENV_PLACEHOLDER
            return env_value
        return _ENV_PATTERN.sub(_replace_env, obj)
    return obj


def _replace_env(match: re.Match) -> str:
    var = match.group(1)
    value = os.getenv(var)
    if not value:
        default_value = _ENV_DEFAULTS.get(var)
        if default_value is not None:
            return default_value
        if var in _SILENT_MISSING_ENV_VARS:
            return match.group(0)
        return match.group(0)
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_deepsearch_service_config_path() -> Path:
    raw = os.getenv("DEEPSEARCH_SERVICE_CONFIG_PATH") or "config/json_configs/deepsearch_service.json"
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path


def _load_deepsearch_service_config(path: Path) -> DeepSearchServiceConfig:
    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    substituted = _substitute_env(data)
    return DeepSearchServiceConfig.model_validate(substituted)


def _merge_tool_manager_payload(
    *,
    service_tool_manager: ToolManagerConfig | None,
    overrides: ToolManagerConfig | None,
) -> Dict[str, Any]:
    if service_tool_manager is None:
        if overrides is None:
            raise ValueError(
                "tool_manager config is required when inherit_from_deepsearch_service=false or DeepSearch service config is unavailable."
            )
        return overrides.model_dump()

    base = service_tool_manager.model_dump()
    if overrides is None:
        return base

    fields_set = getattr(overrides, "model_fields_set", None)
    if not isinstance(fields_set, set) or not fields_set:
        return base

    override_payload = overrides.model_dump()
    for key in fields_set:
        base[key] = override_payload.get(key)
    return base


def _resolve_enabled_tools(
    *,
    explicit: Optional[List[str]],
    service_config: DeepSearchServiceConfig | None,
) -> List[str]:
    if explicit:
        tokens = [str(name).strip() for name in explicit if str(name).strip()]
    else:
        env = os.getenv("DEEPSEARCH_TOOL_MCP_TOOLS")
        if env:
            raw = env.strip()
            if raw.lower() in {"__all__", "all", "*"}:
                tokens = [descriptor.name for descriptor in builtin_tool_descriptors()]
            else:
                tokens = [item.strip() for item in raw.split(",") if item.strip()]
        else:
            allowlist = None
            if service_config is not None:
                allowlist = getattr(getattr(service_config.graph_reasoning, "think", None), "tool_catalog_allowlist", None)
            tokens = [str(name).strip() for name in (allowlist or []) if str(name).strip()]
            think_tool = None
            if service_config is not None:
                think_tool = getattr(getattr(service_config.graph_reasoning, "think", None), "tool_name", None)
            if think_tool:
                tokens.append(str(think_tool).strip())

    builtin = {descriptor.name for descriptor in builtin_tool_descriptors()}
    filtered = [name for name in tokens if name in builtin]
    # Keep stable ordering and uniqueness.
    seen: set[str] = set()
    ordered: List[str] = []
    for name in filtered:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    if not ordered:
        raise ValueError("Enabled tool list resolved to empty. Set enabled_tools or DEEPSEARCH_TOOL_MCP_TOOLS.")
    return ordered
