"""Tool manager that orchestrates local registries and MCP routing (infrastructure layer)."""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote, ToolExecutionLog, ToolResultPayload, GraphQueryContext
from encapsulation.mcp.client import MCPToolClient

from core.deepsearch.tools import (
    GraphTool,
    ToolDescriptor,
    ToolRunRequest,
    builtin_tool_descriptors,
    build_builtin_tools,
    get_tool_descriptor,
)

from core.deepsearch.tooling.registry import DEFAULT_TOOL_HINT_REGISTRY, ToolHintRegistry
from core.utils.json_safe import json_safe
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.deepsearch.trace import emit_trace

logger = logging.getLogger(__name__)


class LocalToolRegistry:
    """Stores instantiated local tools and descriptor metadata."""

    def __init__(
        self,
        *,
        tool_configs: Optional[Dict[str, Any]] = None,
        llm_connector=None,
        injected_tools: Optional[Dict[str, GraphTool]] = None,
        tool_hint_registry: ToolHintRegistry | None = None,
    ):
        self.tool_configs = tool_configs or {}
        self.tool_hint_registry = tool_hint_registry or DEFAULT_TOOL_HINT_REGISTRY
        self.audit_label = self.tool_configs.get("audit_label")
        self._enabled_tool_configs = self.tool_configs.get("enabled_tools") or {}
        builtin_map = {desc.name: desc for desc in builtin_tool_descriptors()}
        self._builtin_tool_names = set(builtin_map.keys())
        self._descriptors = dict(builtin_map)
        self._register_remote_descriptors()
        self._tools: Dict[str, GraphTool] = {}
        if self._is_enabled_globally():
            overrides = self._build_overrides()
            connector = self._coerce_llm_connector(llm_connector)
            builtin = build_builtin_tools(llm_connector=connector, overrides=overrides)
            for name, tool in builtin.items():
                if self._tool_enabled(name):
                    self._tools[name] = tool
        if injected_tools:
            for name, tool in injected_tools.items():
                self._tools[name] = tool
                self._register_custom_descriptor(name, tool)
        self._update_disabled_hints()

    def resolve(self, tool_name: str) -> GraphTool | None:
        """Return the instantiated tool when available."""

        return self._tools.get(tool_name)

    def descriptor_for(self, tool_name: str) -> ToolDescriptor | None:
        """Return the descriptor either from the tool instance or built-in map."""

        tool = self._tools.get(tool_name)
        if tool and getattr(tool, "descriptor", None):
            return tool.descriptor  # type: ignore[attr-defined]
        return self._descriptors.get(tool_name)

    def should_route_remote(self, tool_name: str) -> bool:
        """Return True when config requests MCP fallback."""

        cfg = self._tool_config(tool_name)
        return bool(cfg.get("mcp_fallback") or cfg.get("mcp_only"))

    def _is_enabled_globally(self) -> bool:
        flag = self.tool_configs.get("enable_builtin_tools", True)
        return bool(flag) if isinstance(flag, bool) else True

    def _tool_config(self, tool_name: str) -> Dict[str, Any]:
        cfg = self._enabled_tool_configs.get(tool_name)
        return cfg if isinstance(cfg, dict) else {}

    def _tool_enabled(self, tool_name: str) -> bool:
        cfg = self._tool_config(tool_name)
        if not cfg:
            return True
        if cfg.get("mcp_only"):
            return False
        if "enabled" in cfg:
            return bool(cfg["enabled"])
        return True

    def is_mcp_only(self, tool_name: str) -> bool:
        cfg = self._tool_config(tool_name)
        return bool(cfg.get("mcp_only"))

    def _build_overrides(self) -> Dict[str, Dict[str, Any]]:
        overrides: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._enabled_tool_configs.items():
            if not isinstance(cfg, dict):
                continue
            params = dict(cfg.get("params") or {})
            if not params:
                params = {
                    key: value
                    for key, value in cfg.items()
                    if key not in {"enabled", "audit_label", "mcp_only", "mcp_fallback"}
                }
            if params:
                overrides[name] = params
        return overrides

    def _register_remote_descriptors(self) -> None:
        remote_tools = self.tool_configs.get("remote_tools") or {}
        for name, raw in remote_tools.items():
            spec = self._normalize_remote_descriptor(raw)
            if not spec:
                continue
            extra_kwargs: Dict[str, Any] = {}
            if spec.get("input_schema"):
                extra_kwargs["input_schema"] = spec["input_schema"]
            if spec.get("example_args"):
                extra_kwargs["example_args"] = spec["example_args"]
            descriptor = ToolDescriptor(
                name=name,
                channel=spec.get("channel", "graph"),
                description=spec["description"],
                speed=spec.get("speed", "medium"),
                cost=spec.get("cost", "medium"),
                strategy_tags=tuple(spec.get("strategy_tags", [])),
                profile=spec.get("profile", "F"),
                determinism=spec.get("determinism", "deterministic"),
                namespace=spec.get("namespace", name),
                mcp_callable=True,
                **extra_kwargs,
            )
            self._descriptors[name] = descriptor
            self.tool_hint_registry.register_tool_hints([descriptor.as_hint()])

    @staticmethod
    def _normalize_remote_descriptor(raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if hasattr(raw, "model_dump"):
            return raw.model_dump()
        if isinstance(raw, dict):
            return raw
        return None

    @staticmethod
    def _coerce_llm_connector(candidate):
        """Return connector only when it exposes chat/achat interfaces."""

        if candidate is None:
            return None
        for attr in ("achat", "chat"):
            handle = getattr(candidate, attr, None)
            if callable(handle):
                return candidate
        return None

    def _update_disabled_hints(self) -> None:
        disabled: set[str] = set()
        if not self._is_enabled_globally():
            for name in self._builtin_tool_names:
                cfg = self._tool_config(name)
                if cfg.get("mcp_fallback") or cfg.get("mcp_only"):
                    continue
                disabled.add(name)
        for name, cfg in self._enabled_tool_configs.items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("enabled") is False and not (cfg.get("mcp_fallback") or cfg.get("mcp_only")):
                disabled.add(name)
        self.tool_hint_registry.set_disabled_tools(disabled)

    def _register_custom_descriptor(self, name: str, tool: GraphTool) -> None:
        descriptor = getattr(tool, "descriptor", None)
        if not descriptor:
            return
        self._descriptors[name] = descriptor
        self.tool_hint_registry.register_tool_hints([descriptor.as_hint()])


@dataclass
class MCPToolRouter:
    """Routes tool calls to MCP servers."""

    mcp_client: Optional[MCPToolClient]
    default_server_name: Optional[str] = None

    async def invoke(
        self,
        descriptor: ToolDescriptor,
        payload: Dict[str, Any],
    ) -> Tuple[ToolResultPayload, ToolExecutionLog]:
        if not self.mcp_client:
            raise RuntimeError("MCPToolRouter cannot invoke tools without an MCP client")
        graph_context = payload.get("graph_context")
        arguments = payload.get("arguments", {})
        server_name = payload.get("server_name") or self.default_server_name
        outcome = await self.mcp_client.call_tool(
            descriptor.namespace or descriptor.name,
            arguments=arguments,
            graph_context=graph_context,
            server_name=server_name,
        )
        payload_model = self._normalize_result(descriptor, outcome.result)
        payload_model.diagnostics.setdefault("latency_ms", outcome.log.latency_ms)
        payload_model.diagnostics.setdefault("transport", outcome.log.extra.get("transport"))
        return payload_model, outcome.log

    def _normalize_result(self, descriptor: ToolDescriptor, result: Any) -> ToolResultPayload:
        content = getattr(result, "content", None) or []
        structured = self._extract_structured_content(content)
        if structured:
            return self._payload_from_structured(descriptor, structured)
        return self._payload_from_text(descriptor, content)

    @staticmethod
    def _extract_structured_content(content: List[Any]) -> Optional[Dict[str, Any]]:
        import json

        for block in content:
            payload = getattr(block, "json", None) or getattr(block, "data", None)
            if isinstance(payload, dict):
                return payload
            text = getattr(block, "text", None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
        return None

    def _payload_from_structured(self, descriptor: ToolDescriptor, data: Dict[str, Any]) -> ToolResultPayload:
        evidences = self._coerce_evidences(descriptor, data.get("evidences", []))
        think_notes = self._coerce_think_notes(data.get("think_notes", []))
        diagnostics = data.get("diagnostics") or {}
        summary = data.get("summary") or data.get("text") or data.get("thought") or ""
        if not think_notes and data.get("thought"):
            next_actions = data.get("next_actions")
            if isinstance(next_actions, list):
                parsed_actions = [str(item) for item in next_actions if str(item).strip()]
            else:
                parsed_actions = []
            plan_step_ref = data.get("plan_step") or data.get("plan_step_id") or data.get("plan_id")
            if plan_step_ref is not None:
                plan_step_ref = str(plan_step_ref)
            think_notes = [
                ThinkNote(
                    plan_step_id=plan_step_ref,
                    reasoning=str(data.get("thought")),
                    confidence_delta=data.get("confidence_delta"),
                    coverage_delta=data.get("coverage_delta"),
                    next_actions=parsed_actions,
                    metadata={"raw": data},
                )
            ]
        return ToolResultPayload(
            tool_name=descriptor.name,
            namespace=descriptor.namespace,
            channel=descriptor.channel,
            profile=descriptor.profile,
            determinism=descriptor.determinism,
            summary=str(summary),
            evidences=evidences,
            diagnostics=diagnostics,
            think_notes=think_notes,
        )

    def _payload_from_text(self, descriptor: ToolDescriptor, content: List[Any]) -> ToolResultPayload:
        evidences: List[EvidenceChunk] = []
        for idx, block in enumerate(content):
            text = getattr(block, "text", None)
            if not text:
                continue
            normalized = str(text).strip()
            evidences.append(
                EvidenceChunk(
                    chunk_id=hashed_chunk_id(source=descriptor.namespace or descriptor.name, content=normalized, prefix="mcp"),
                    source=descriptor.namespace or descriptor.name,
                    content=normalized,
                    provenance={"content_type": getattr(block, "type", "text")},
                )
            )
        summary = evidences[0].content if evidences else ""
        return ToolResultPayload(
            tool_name=descriptor.name,
            namespace=descriptor.namespace,
            channel=descriptor.channel,
            profile=descriptor.profile,
            determinism=descriptor.determinism,
            summary=summary,
            evidences=evidences,
            diagnostics={},
            think_notes=[],
        )

    @staticmethod
    def _coerce_evidences(descriptor: ToolDescriptor, payload: Any) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        if not isinstance(payload, list):
            return evidences
        for idx, item in enumerate(payload):
            if isinstance(item, EvidenceChunk):
                evidences.append(item)
                continue
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("chunk_id") or hashed_chunk_id(
                source=str(item.get("source") or descriptor.namespace or descriptor.name),
                content=str(item.get("content") or ""),
                prefix="mcp",
            )
            source = item.get("source") or descriptor.namespace or descriptor.name
            content = item.get("content")
            if not content:
                continue
            evidences.append(
                EvidenceChunk(
                    chunk_id=str(chunk_id),
                    source=str(source),
                    content=str(content),
                    score=item.get("score"),
                    provenance=item.get("provenance") or {},
                )
            )
        return evidences

    @staticmethod
    def _coerce_think_notes(payload: Any) -> List[ThinkNote]:
        notes: List[ThinkNote] = []
        if not isinstance(payload, list):
            return notes
        for item in payload:
            if isinstance(item, ThinkNote):
                notes.append(item)
                continue
            if isinstance(item, dict):
                try:
                    notes.append(ThinkNote(**item))
                except Exception:
                    continue
        return notes


class DeepSearchToolManager:
    """Handles local tool invocation, optional MCP calls, and telemetry."""

    def __init__(
        self,
        tool_configs: Optional[Dict[str, Any]],
        telemetry_client,
        *,
        mcp_client: MCPToolClient | None = None,
        local_tools: Optional[Dict[str, GraphTool]] = None,
        local_registry: Optional[LocalToolRegistry] = None,
        mcp_router: Optional[MCPToolRouter] = None,
        tool_hint_registry: ToolHintRegistry | None = None,
    ):
        self.tool_configs = tool_configs or {}
        self.telemetry_client = telemetry_client
        self.local_registry = local_registry or LocalToolRegistry(
            tool_configs=self.tool_configs,
            llm_connector=self.tool_configs.get("llm_connector"),
            injected_tools=local_tools,
            tool_hint_registry=tool_hint_registry,
        )
        self.mcp_router = mcp_router or MCPToolRouter(
            mcp_client=mcp_client,
            default_server_name=self.tool_configs.get("default_mcp_server"),
        )
        self.remote_argument_templates = self.tool_configs.get("remote_argument_templates") or {}
        self.audit_label = self.tool_configs.get("audit_label") or getattr(self.local_registry, "audit_label", None)
        self.max_remote_evidences = int(self.tool_configs.get("max_remote_evidences", 32))
        self.max_remote_context_chars = int(self.tool_configs.get("max_remote_context_chars", 4096))
        self.llm_fingerprint = self._fingerprint_llm(self.tool_configs.get("llm_connector"))
        artifact_dir = self.tool_configs.get("artifact_dir") or os.getenv("DEEPSEARCH_TOOL_ARTIFACT_DIR")
        self.artifact_dir = Path(artifact_dir).expanduser() if artifact_dir else None

    async def invoke(self, tool_name: str, *, payload: Dict[str, Any]) -> ToolResultPayload:
        """Invoke a tool through MCP first, falling back to local registries on failure."""

        call_id = uuid.uuid4().hex
        descriptor = self._resolve_descriptor(tool_name)
        request = self._build_request(payload)
        local_tool = self.local_registry.resolve(tool_name) if self.local_registry else None
        local_disabled = self._prefer_remote_for(tool_name)
        if local_disabled:
            local_tool = None

        trace_call = {
            "call_id": call_id,
            "tool_name": tool_name,
            "descriptor": (descriptor.as_hint() if descriptor else None),
            "plan_step": request.plan_step,
            "question": request.question,
            "extra": request.extra,
            "coverage_metrics": request.coverage_metrics,
            "access_scope": self._access_scope_payload(request.access_scope),
            "graph_context": (request.graph_context.model_dump(exclude_none=True) if request.graph_context else None),
            "routing": {
                "can_route_remote": bool(self._can_route_remote(tool_name, descriptor)),
                "prefer_remote": bool(local_disabled),
                "has_local": bool(local_tool is not None),
                "default_mcp_server": getattr(self.mcp_router, "default_server_name", None) if self.mcp_router else None,
            },
        }
        await emit_trace(
            "tool_call",
            json.dumps(json_safe(trace_call), ensure_ascii=False, indent=2, default=str),
            meta={"call_id": call_id, "tool_name": tool_name, "plan_step": request.plan_step},
        )

        remote_error: Exception | None = None
        if self._can_route_remote(tool_name, descriptor):
            try:
                result = await self._invoke_remote(tool_name, descriptor, payload, request)
                await emit_trace(
                    "tool_response",
                    json.dumps(
                        json_safe(
                            {
                                "call_id": call_id,
                                "tool_name": tool_name,
                                "route": "remote",
                                "result": result.model_dump(exclude_none=True),
                            }
                        ),
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    meta={
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "plan_step": request.plan_step,
                        "ok": True,
                        "route": "remote",
                    },
                )
                return result
            except Exception as exc:  # noqa: BLE001
                remote_error = exc
                logger.warning(
                    "Remote tool %s failed via MCP (%s); attempting local fallback",
                    tool_name,
                    exc,
                )

        if local_tool:
            result = await self._invoke_local(tool_name, local_tool, descriptor, request)
            if remote_error:
                result.diagnostics.setdefault("remote_fallback_reason", str(remote_error))
            await emit_trace(
                "tool_response",
                json.dumps(
                    json_safe(
                        {
                            "call_id": call_id,
                            "tool_name": tool_name,
                            "route": "local",
                            "remote_fallback_reason": (str(remote_error) if remote_error else None),
                            "result": result.model_dump(exclude_none=True),
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                meta={
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "plan_step": request.plan_step,
                    "ok": True,
                    "route": "local",
                    "remote_fallback": bool(remote_error is not None),
                },
            )
            return result

        if remote_error:
            await emit_trace(
                "tool_response",
                json.dumps(
                    json_safe(
                        {
                            "call_id": call_id,
                            "tool_name": tool_name,
                            "route": "remote",
                            "error": str(remote_error),
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                meta={
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "plan_step": request.plan_step,
                    "ok": False,
                    "route": "remote",
                },
            )
            raise remote_error

        await emit_trace(
            "tool_response",
            json.dumps(
                json_safe(
                    {
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "error": "tool_unavailable",
                    }
                ),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            meta={"call_id": call_id, "tool_name": tool_name, "plan_step": request.plan_step, "ok": False},
        )
        raise KeyError(f"Tool '{tool_name}' is not registered locally and MCP routing is unavailable")

    async def _invoke_remote(
        self,
        tool_name: str,
        descriptor: ToolDescriptor,
        payload: Dict[str, Any],
        request: ToolRunRequest,
    ) -> ToolResultPayload:
        remote_payload = self._prepare_remote_payload(tool_name, payload, request)
        payload_model, log = await self.mcp_router.invoke(descriptor, remote_payload)  # type: ignore[arg-type]
        self._attach_artifact_reference(tool_name, payload_model)
        self._record_remote(tool_name, log, request, payload_model, descriptor)
        return payload_model

    async def _invoke_local(
        self,
        tool_name: str,
        tool: GraphTool,
        descriptor: ToolDescriptor | None,
        request: ToolRunRequest,
    ) -> ToolResultPayload:
        start = time.perf_counter()
        result = await tool.run(request)
        latency_ms = int((time.perf_counter() - start) * 1000)
        descriptor = descriptor or getattr(tool, "descriptor", None)
        if descriptor is None:
            raise RuntimeError(f"Local tool '{tool_name}' does not expose a descriptor")
        payload_model = result.as_payload(descriptor)
        payload_model.diagnostics.setdefault("latency_ms", latency_ms)
        payload_model.diagnostics.setdefault("evidence_count", len(payload_model.evidences or []))
        self._attach_artifact_reference(tool_name, payload_model)
        self._record_local(tool_name, payload_model, request, latency_ms, descriptor)
        return payload_model

    def _resolve_descriptor(self, tool_name: str) -> ToolDescriptor | None:
        descriptor = None
        if self.local_registry:
            descriptor = self.local_registry.descriptor_for(tool_name)
        if descriptor:
            return descriptor
        return get_tool_descriptor(tool_name)

    def _prefer_remote_for(self, tool_name: str) -> bool:
        if not self.local_registry:
            return False
        return self.local_registry.is_mcp_only(tool_name)

    def _can_route_remote(self, tool_name: str, descriptor: ToolDescriptor | None) -> bool:
        if not descriptor or not self.mcp_router or not getattr(self.mcp_router, "mcp_client", None):
            return False
        if descriptor.mcp_callable:
            return True
        return bool(self.local_registry and self.local_registry.should_route_remote(tool_name))

    def _prepare_remote_payload(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        request: ToolRunRequest,
    ) -> Dict[str, Any]:
        arguments = self._build_remote_arguments(tool_name, request, payload)
        return {
            "arguments": arguments,
            "graph_context": request.graph_context,
            "server_name": payload.get("server_name"),
        }

    def _build_remote_arguments(
        self,
        tool_name: str,
        request: ToolRunRequest,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        context_window = self._serialize_context_window(request)
        arguments: Dict[str, Any] = {
            "question": request.question,
            "plan_step": request.plan_step,
            "context_evidences": context_window,
            "extra": self._json_safe(request.extra),
            "adapter_metadata": self._json_safe(self._adapter_metadata(request.adapter)),
            "access_scope": self._access_scope_payload(request.access_scope),
        }
        arguments = {key: value for key, value in arguments.items() if value is not None}
        digest = self._context_digest(context_window)
        if digest:
            arguments["context_digest"] = digest
        if request.graph_context:
            arguments["graph_context"] = self._json_safe(
                request.graph_context.model_dump(exclude_none=True)
            )
        if request.coverage_metrics:
            arguments["coverage_metrics"] = self._json_safe(request.coverage_metrics)
        template_args = self._render_argument_template(tool_name, request, payload)
        if template_args:
            arguments.update(template_args)
        return arguments

    def _render_argument_template(
        self,
        tool_name: str,
        request: ToolRunRequest,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        template = self.remote_argument_templates.get(tool_name)
        if not isinstance(template, dict):
            return {}
        context = self._build_template_context(request, payload)
        rendered: Dict[str, Any] = {}
        for key, value in template.items():
            if isinstance(value, str):
                rendered[key] = Template(value).safe_substitute(context)
            else:
                rendered[key] = value
        return rendered

    @staticmethod
    def _build_template_context(request: ToolRunRequest, payload: Dict[str, Any]) -> Dict[str, Any]:
        context = {
            "question": request.question,
            "plan_step": request.plan_step or "",
            "extra": request.extra,
            "payload": payload,
        }
        if request.graph_context:
            context["graph_context"] = request.graph_context.model_dump(exclude_none=True)
        if request.coverage_metrics:
            context["coverage_metrics"] = request.coverage_metrics
        return context

    @staticmethod
    def _adapter_metadata(adapter) -> Optional[Dict[str, Any]]:
        if adapter is None:
            return None
        meta = getattr(adapter, "metadata", None)
        if not callable(meta):
            return None
        metadata = meta()
        try:
            return asdict(metadata)
        except Exception:
            return metadata.__dict__

    @staticmethod
    def _access_scope_payload(access_scope) -> Optional[Dict[str, Any]]:
        if access_scope is None:
            return None
        payload = {
            "scope_id": getattr(access_scope, "scope_id", None),
            "scope_type": getattr(access_scope, "scope_type", None),
            "labels": list(getattr(access_scope, "labels", []) or []),
            "attributes": getattr(access_scope, "attributes", None),
        }
        return {k: v for k, v in payload.items() if v is not None}

    @staticmethod
    def _build_request(payload: Dict[str, Any]) -> ToolRunRequest:
        context_payload = payload.get("context_evidences") or []
        context_evidences: List[EvidenceChunk] = []
        for item in context_payload:
            if isinstance(item, EvidenceChunk):
                context_evidences.append(item)
            elif isinstance(item, dict):
                try:
                    context_evidences.append(EvidenceChunk(**item))
                except Exception:
                    continue
        adapter = payload.get("adapter")
        access_scope = payload.get("access_scope")
        if access_scope and not hasattr(access_scope, "as_token"):
            from core.graph_adapter.base import GraphAccessScope

            access_scope = GraphAccessScope(**access_scope)
        graph_context = payload.get("graph_context")
        if graph_context and isinstance(graph_context, dict):
            graph_context = GraphQueryContext(**graph_context)
        return ToolRunRequest(
            question=payload.get("question", ""),
            plan_step=payload.get("plan_step"),
            context_evidences=context_evidences,
            adapter=adapter,
            access_scope=access_scope,
            extra=payload.get("extra", {}),
            graph_context=graph_context,
            coverage_metrics=payload.get("coverage_metrics"),
        )

    def _record_local(
        self,
        tool_name: str,
        result: ToolResultPayload,
        request: ToolRunRequest,
        latency_ms: int,
        descriptor: ToolDescriptor | None,
    ) -> None:
        if not self.telemetry_client:
            return
        log_method = getattr(self.telemetry_client, "log_tool_invocation", None)
        if not callable(log_method):
            return
        payload = {
            "run_id": self._extract_run_id(request),
            "summary": result.summary,
            "diagnostics": result.diagnostics,
            "question": request.question,
            "plan_step": request.plan_step,
            "think_notes": [note.model_dump() for note in result.think_notes],
            "latency_ms": latency_ms,
            "evidence_count": len(result.evidences or []),
            "adapter_metadata": self._adapter_metadata(request.adapter),
            "tool_namespace": descriptor.namespace if descriptor else None,
            "external_allowed": self._extract_external_allowed(request),
        }
        if self.llm_fingerprint:
            payload["llm_fingerprint"] = self.llm_fingerprint
        if self.audit_label:
            payload["audit_label"] = self.audit_label
        log_method(
            tool_name=tool_name,
            payload=payload,
        )

    def _record_remote(
        self,
        tool_name: str,
        log: ToolExecutionLog,
        request: ToolRunRequest | None = None,
        result: ToolResultPayload | None = None,
        descriptor: ToolDescriptor | None = None,
    ) -> None:
        if not self.telemetry_client:
            return
        if self.audit_label:
            log.extra.setdefault("audit_label", self.audit_label)
        log.extra.setdefault("run_id", self._extract_run_id(request) if request else None)
        if request and log.graph_context is None and request.graph_context:
            log.graph_context = request.graph_context
        if request:
            adapter_meta = self._adapter_metadata(request.adapter)
            if adapter_meta:
                log.extra.setdefault("adapter_metadata", adapter_meta)
            log.extra.setdefault("external_allowed", self._extract_external_allowed(request))
        if self.llm_fingerprint:
            log.extra.setdefault("llm_fingerprint", self.llm_fingerprint)
        if descriptor:
            log.extra.setdefault("tool_namespace", descriptor.namespace)
        if result is not None:
            log.extra.setdefault("evidence_count", len(result.evidences or []))
        log_method = getattr(self.telemetry_client, "log_remote_tool", None)
        if callable(log_method):
            log_method(tool_name=tool_name, log=log)

    @staticmethod
    def _extract_run_id(request: ToolRunRequest | None) -> Optional[str]:
        if not request or not request.graph_context:
            return None
        metadata = request.graph_context.metadata or {}
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("run_id") or (metadata.get("request_metadata") or {}).get("run_id")
        return str(value) if value else None

    @staticmethod
    def _extract_external_allowed(request: ToolRunRequest | None) -> Optional[bool]:
        if not request or not request.graph_context:
            return None
        metadata = request.graph_context.metadata or {}
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("external_allowed")
        return bool(value) if isinstance(value, bool) else None

    def _serialize_context_window(self, request: ToolRunRequest) -> List[Dict[str, Any]]:
        if not request.context_evidences:
            return []
        window = request.context_evidences
        limit = max(0, self.max_remote_evidences)
        if limit:
            window = window[-limit:]
        char_budget = self.max_remote_context_chars if self.max_remote_context_chars > 0 else None
        serialized: List[Dict[str, Any]] = []
        for evidence in window:
            snippet = (evidence.content or "").strip()
            if char_budget is not None:
                take = min(len(snippet), char_budget)
                snippet = snippet[:take]
                char_budget -= take
                if char_budget <= 0:
                    char_budget = 0
            serialized.append(
                {
                    "chunk_id": evidence.chunk_id,
                    "source": evidence.source,
                    "score": evidence.score,
                    "content": snippet,
                }
            )
        return serialized

    @staticmethod
    def _context_digest(serialized_window: List[Dict[str, Any]]) -> Optional[str]:
        if not serialized_window:
            return None
        snippets: List[str] = []
        for entry in serialized_window:
            snippet = (entry.get("content") or "").strip()
            if not snippet:
                continue
            chunk_id = entry.get("chunk_id") or "chunk"
            snippets.append(f"[{chunk_id}] {snippet}")
        return "\n".join(snippets) if snippets else None

    @staticmethod
    def _fingerprint_llm(connector) -> Optional[str]:
        if connector is None:
            return None
        if isinstance(connector, str):
            return connector
        for attr in ("model_name", "model_id", "model"):
            value = getattr(connector, attr, None)
            if isinstance(value, str) and value:
                return value
        return connector.__class__.__name__

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return json_safe(value)

    def _attach_artifact_reference(self, tool_name: str, payload: ToolResultPayload) -> None:
        artifact_path = self._persist_artifact(tool_name, payload)
        if not artifact_path:
            return
        diagnostics = payload.diagnostics
        artifacts = diagnostics.setdefault("artifacts", [])
        artifacts.append({"type": "file", "path": artifact_path})

    def _persist_artifact(self, tool_name: str, payload: ToolResultPayload) -> Optional[str]:
        if not self.artifact_dir:
            return None
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create artifact directory %s: %s", self.artifact_dir, exc)
            return None
        file_name = f"{int(time.time() * 1000)}_{tool_name.replace('.', '_')}_{uuid.uuid4().hex}.json"
        artifact_path = self.artifact_dir / file_name
        try:
            with artifact_path.open("w", encoding="utf-8") as handle:
                json.dump(payload.model_dump(), handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Failed to persist tool artifact for %s: %s", tool_name, exc)
            return None
        return str(artifact_path)
