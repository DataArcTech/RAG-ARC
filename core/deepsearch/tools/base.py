"""Common primitives shared by DeepSearch tools."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote, ToolResultPayload, GraphQueryContext
from core.graph_adapter.base import GraphAccessScope, GraphDeepSearchAdapter


def _default_input_schema() -> Dict[str, Any]:
    """Return the generic MCP schema shared by graph tools."""

    return {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Current natural language question."},
            "plan_step": {"type": "string", "description": "Identifier of the originating plan step."},
            "context_evidences": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Evidence chunks accumulated so far.",
            },
            "extra": {"type": "object", "description": "Tool-specific metadata or hints."},
            "adapter_metadata": {"type": "object", "description": "Graph adapter metadata and capabilities."},
            "access_scope": {"type": "object", "description": "Structured tenant/user scope."},
            "graph_context": {"type": "object", "description": "Serialized GraphQueryContext snapshot."},
            "coverage_metrics": {"type": "object", "description": "Optional coverage/confidence diagnostics."},
        },
        "required": ["question"],
        "additionalProperties": True,
    }


def build_input_schema(
    *,
    extra_properties: Optional[Dict[str, Any]] = None,
    required_fields: Optional[Iterable[str]] = None,
    required_extra_fields: Optional[Iterable[str]] = None,
    top_level_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a customised schema with extra fields for a specific tool."""

    schema = deepcopy(_default_input_schema())
    properties = schema["properties"]
    if top_level_overrides:
        properties.update(top_level_overrides)
    if required_fields:
        required = list(schema.get("required", []))
        for field in required_fields:
            if field and field not in required:
                required.append(field)
        schema["required"] = required
    if extra_properties or required_extra_fields:
        extra_schema = dict(properties.get("extra") or {"type": "object"})
        extra_props = extra_schema.setdefault("properties", {})
        if extra_properties:
            extra_props.update(extra_properties)
        if required_extra_fields:
            required = list(extra_schema.get("required", []))
            for field in required_extra_fields:
                if field and field not in required:
                    required.append(field)
            if required:
                extra_schema["required"] = required
        properties["extra"] = extra_schema
    return schema


@dataclass(frozen=True)
class ToolDescriptor:
    """Metadata describing a tool for planner hints and observability."""

    name: str
    channel: str
    description: str
    speed: str = "medium"
    cost: str = "medium"
    strategy_tags: Iterable[str] = field(default_factory=tuple)
    profile: str = "F"
    determinism: str = "deterministic"
    namespace: str = "rag-arc.deepsearch.tools"
    mcp_callable: bool = False
    input_schema: Dict[str, Any] = field(default_factory=_default_input_schema)
    example_args: Optional[Dict[str, Any]] = None

    def as_hint(self) -> Dict[str, str]:
        """Return a dict representation consumed by planner prompts."""

        return {
            "name": self.name,
            "channel": self.channel,
            "description": self.description,
            "profile": self.profile,
            "determinism": self.determinism,
            "namespace": self.namespace,
            "speed": self.speed,
            "cost": self.cost,
            "strategy_tags": list(self.strategy_tags),
        }

    def as_mcp_spec(self) -> Dict[str, Any]:
        """Return a dict that can be passed to MCP Tool registration."""

        payload = {
            "name": self.namespace or self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        if self.example_args:
            payload["examples"] = [self.example_args]
        return payload


@dataclass
class ToolRunRequest:
    """Runtime payload handed to tool implementations."""

    question: str
    plan_step: Optional[str]
    context_evidences: List[EvidenceChunk]
    adapter: Optional[GraphDeepSearchAdapter]
    access_scope: Optional[GraphAccessScope]
    extra: Dict[str, Any] = field(default_factory=dict)
    graph_context: Optional[GraphQueryContext] = None
    coverage_metrics: Optional[Dict[str, Any]] = None


@dataclass
class ToolResult:
    """Normalized tool output consumed by GraphReasoning or reporters."""

    summary: str
    evidences: List[EvidenceChunk] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    think_notes: List[ThinkNote] = field(default_factory=list)

    def as_payload(self, descriptor: ToolDescriptor) -> ToolResultPayload:
        """Convert the internal representation into a serializable payload."""

        return ToolResultPayload(
            tool_name=descriptor.name,
            namespace=descriptor.namespace,
            channel=descriptor.channel,
            profile=descriptor.profile,
            determinism=descriptor.determinism,
            summary=self.summary,
            evidences=self.evidences,
            diagnostics=self.diagnostics,
            think_notes=self.think_notes,
        )


class GraphTool(Protocol):
    """Protocol implemented by every DeepSearch graph-aware tool."""

    descriptor: ToolDescriptor

    async def run(self, request: ToolRunRequest) -> ToolResult:
        """Execute the tool logic and return a structured result."""


async def call_llm_async(llm, messages: List[Dict[str, str]], **kwargs) -> str:
    """Utility to invoke sync/async LLM connectors transparently."""

    if llm is None:
        raise RuntimeError("LLM connector is required for this tool")

    async_chat = getattr(llm, "achat", None)
    if callable(async_chat):
        return await async_chat(messages, **kwargs)

    chat = getattr(llm, "chat", None)
    if not callable(chat):
        raise RuntimeError("LLM connector does not expose chat/achat methods")
    return chat(messages, **kwargs)
