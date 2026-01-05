"""Tool that compares multiple graph adapters and suggests coordination plans."""
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Sequence

from encapsulation.data_model.deepsearch import EvidenceChunk

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, build_input_schema, safe_json_loads
from core.deepsearch.utils.evidence_ids import derived_chunk_id


class CrossAdapterPlannerTool(GraphTool):
    """Produces coordinated plans when multiple adapters are available."""

    descriptor = ToolDescriptor(
        name="graph.cross_adapter_planner",
        channel="graph",
        description="Summarises adapter trade-offs and emits JSON `{summary, actions[]}` plans so downstream "
        "steps know which graph adapter should handle each hop or validation task.",
        speed="slow",
        cost="high",
        strategy_tags=("planner", "meta_reasoning", "llm"),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.heavy.cross_adapter_planner",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "alternate_adapters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "adapter_name": {"type": "string"},
                            "graph_type": {"type": "string"},
                            "version": {"type": "string"},
                        },
                        "required": ["adapter_name", "graph_type", "version"],
                    },
                    "description": "Alternate adapters to include in the orchestration plan.",
                }
            }
        ),
        example_args={
            "question": "Compare HippoRAG and LightRAG",
            "plan_step": "plan_meta",
            "extra": {
                "alternate_adapters": [
                    {"adapter_name": "hipporag", "graph_type": "hipporag", "version": "1.0"},
                    {"adapter_name": "lightrag", "graph_type": "lightrag", "version": "1.0"},
                ]
            },
        },
    )

    def __init__(self, llm_connector, *, temperature: float = 0.2):
        self.llm_connector = llm_connector
        self.temperature = temperature

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapters = self._collect_adapters(request)
        if not adapters:
            return ToolResult(summary="Cross-adapter planner skipped: no alternate adapters provided.")

        summary_text, plan_items = await self._generate_plan(request, adapters)
        evidences = [
            EvidenceChunk(
                chunk_id=derived_chunk_id(
                    tool_name=self.descriptor.name,
                    plan_step=request.plan_step,
                    label=f"adapter_plan_{idx}",
                    content=str(item),
                ),
                source=self.descriptor.name,
                content=item,
                provenance={"adapters": adapters},
            )
            for idx, item in enumerate(plan_items)
        ]
        diagnostics = {
            "adapter_count": len(adapters),
            "thought_log": self._build_thought_log(adapters, plan_items, request.plan_step),
        }
        return ToolResult(summary=summary_text, evidences=evidences, diagnostics=diagnostics)

    def _collect_adapters(self, request: ToolRunRequest) -> List[Dict[str, Any]]:
        adapters: List[Dict[str, Any]] = []
        if request.adapter:
            adapters.append(self._metadata_to_dict(request.adapter.metadata()))
        for alt in request.extra.get("alternate_adapters", []):
            adapters.append(self._metadata_to_dict(alt))
        return [adapter for adapter in adapters if adapter]

    async def _generate_plan(
        self,
        request: ToolRunRequest,
        adapters: List[Dict[str, Any]],
    ) -> tuple[str, List[str]]:
        payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "adapters": adapters,
        }
        messages = [
            {
                "role": "system",
                "content": "Compare adapters and output JSON with 'summary' and 'actions'.",
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
        data = safe_json_loads(response, expected="dict")
        if not isinstance(data, dict):
            raise ValueError("Cross-adapter planner returned non-JSON output")
        summary = str(data.get("summary") or "").strip()
        actions = [str(item) for item in data.get("actions", []) if str(item).strip()]
        if not summary or not actions:
            raise ValueError("Cross-adapter planner returned an incomplete plan payload")
        return summary, actions

    def _metadata_to_dict(self, metadata: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if metadata is None:
            return payload
        if isinstance(metadata, Mapping):
            payload = dict(metadata)
        elif hasattr(metadata, "model_dump"):
            payload = metadata.model_dump()
        elif is_dataclass(metadata):
            payload = asdict(metadata)
        else:
            payload = getattr(metadata, "__dict__", {})
        return self._sanitize(payload)

    def _sanitize(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._sanitize(asdict(value))
        if isinstance(value, Mapping):
            return {str(k): self._sanitize(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize(item) for item in value]
        if hasattr(value, "model_dump"):
            return self._sanitize(value.model_dump())
        return value

    @staticmethod
    def _build_thought_log(
        adapters: List[Dict[str, Any]],
        actions: List[str],
        plan_step: str | None,
    ) -> List[Dict[str, Any]]:
        if not adapters:
            return []
        log: List[Dict[str, Any]] = []
        for idx, action in enumerate(actions):
            log.append(
                {
                    "plan_step": plan_step,
                    "reasoning": action,
                    "reasoning_tags": ["cross_adapter_planner"],
                    "adapter_count": len(adapters),
                    "branch": idx,
                }
            )
        return log
