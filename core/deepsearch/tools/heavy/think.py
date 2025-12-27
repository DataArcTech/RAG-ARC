"""Graph think tool that offers structured reasoning pauses."""
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from encapsulation.data_model.deepsearch import ThinkNote, GraphQueryContext
from core.prompts.deepsearch import THINK_TOOL_SYSTEM_PROMPT

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, safe_json_loads


class ThinkToolCall(BaseModel):
    tool_name: str = Field(..., min_length=1)
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(..., min_length=1)
    parallelizable: bool = Field(...)


class ThinkToolResponse(BaseModel):
    reasoning: str = Field(..., min_length=1)
    confidence_delta: float | None = Field(...)
    coverage_delta: float | None = Field(...)
    next_actions: List[str] = Field(...)
    tool_calls: List[ThinkToolCall] = Field(...)
    gap_trigger: bool = Field(...)
    missing_topics: List[str] = Field(...)


class GraphThinkTool(GraphTool):
    """Implements Anthropic-style think windows tailored for graph reasoning."""

    descriptor = ToolDescriptor(
        name="graph.think",
        channel="graph",
        description="Structured pause that digests current context before the next hop.",
        speed="slow",
        cost="low",
        strategy_tags=("think", "control", "reflection"),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.heavy.think",
        mcp_callable=True,
        example_args={
            "question": "What gaps remain?",
            "plan_step": "plan_02",
            "context_evidences": [],
            "coverage_metrics": {"coverage_score": 0.3, "confidence_score": 0.4},
        },
    )

    def __init__(
        self,
        llm_connector=None,
        *,
        temperature: float = 0.1,
        system_prompt: str = THINK_TOOL_SYSTEM_PROMPT,
    ):
        self.llm_connector = llm_connector
        self.temperature = temperature
        self.system_prompt = system_prompt

    async def run(self, request: ToolRunRequest) -> ToolResult:
        note = await self._build_note(request)
        context_snapshot = self._graph_context_snapshot(request.graph_context)
        coverage_snapshot = request.coverage_metrics or {}
        diagnostics = {
            "plan_step": request.plan_step,
            "temperature": self.temperature,
            "graph_context": context_snapshot,
            "coverage_metrics": coverage_snapshot,
        }
        diagnostics["thought_log"] = [
            self._build_thought_log_entry(
                note,
                coverage_snapshot=coverage_snapshot,
            )
        ]
        summary = note.reasoning
        return ToolResult(summary=summary, diagnostics=diagnostics, think_notes=[note])

    async def _build_note(self, request: ToolRunRequest) -> ThinkNote:
        context_snapshot = self._graph_context_snapshot(request.graph_context)
        coverage_snapshot = request.coverage_metrics or {}
        if not self.llm_connector:
            raise RuntimeError("GraphThinkTool requires an LLM connector")

        prompt_payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "context_evidences": [ev.model_dump(exclude_none=True) for ev in request.context_evidences],
            "graph_context": context_snapshot,
            "coverage_metrics": coverage_snapshot,
            "extra": request.extra,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._serialize_payload(prompt_payload),
            },
        ]
        try:
            response = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
            parsed = safe_json_loads(response, expected="dict")
            if not isinstance(parsed, dict):
                raise ValueError("Think tool returned non-JSON or non-dict payload")
            payload = ThinkToolResponse.model_validate(parsed)
            missing_topics = self._merge_missing_topics(
                payload.missing_topics,
                coverage_snapshot.get("missing_topics"),
            )
            return ThinkNote(
                plan_step_id=request.plan_step,
                reasoning=payload.reasoning,
                confidence_delta=payload.confidence_delta,
                coverage_delta=payload.coverage_delta,
                next_actions=[str(item) for item in payload.next_actions],
                metadata={
                    "raw": parsed,
                    "graph_context": context_snapshot,
                    "coverage_metrics": coverage_snapshot,
                    "gap_trigger": bool(payload.gap_trigger),
                    "missing_topics": missing_topics,
                    "tool_calls": [call.model_dump() for call in payload.tool_calls],
                },
            )
        except Exception as exc:
            raise RuntimeError(f"GraphThinkTool failed: {exc}") from exc

    @staticmethod
    def _graph_context_snapshot(graph_context: GraphQueryContext | None) -> Dict[str, Any]:
        if not graph_context:
            return {}
        return graph_context.model_dump(exclude_none=True)

    @staticmethod
    def _serialize_payload(payload: Dict[str, Any]) -> str:
        def _default(value):
            return str(value)

        return json.dumps(payload, ensure_ascii=False, default=_default)

    @staticmethod
    def _build_thought_log_entry(note: ThinkNote, coverage_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        tags = ["graph.think"]
        coverage_score = coverage_snapshot.get("coverage_score")
        if isinstance(coverage_score, (int, float)) and coverage_score < 0.4:
            tags.append("coverage_gap")
        entry = {
            "plan_step": note.plan_step_id,
            "reasoning": note.reasoning,
            "reasoning_tags": tags,
            "confidence_delta": note.confidence_delta,
            "coverage_delta": note.coverage_delta,
            "latency_ms": coverage_snapshot.get("latency_ms", 0),
        }
        return entry

    @staticmethod
    def _merge_missing_topics(*payloads: Any) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for payload in payloads:
            if isinstance(payload, list):
                for item in payload:
                    token = str(item).strip()
                    if token and token not in seen:
                        seen.add(token)
                        merged.append(token)
        return merged
