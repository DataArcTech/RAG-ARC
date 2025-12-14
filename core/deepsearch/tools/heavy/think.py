"""Graph think tool that offers structured reasoning pauses."""
import json
from typing import Any, Dict, List, Optional

from encapsulation.data_model.deepsearch import ThinkNote, GraphQueryContext
from core.prompts.deepsearch import THINK_TOOL_SYSTEM_PROMPT

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async


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
        heuristic_note = self._heuristic_note(request, coverage_snapshot, context_snapshot)
        if not self.llm_connector:
            return heuristic_note

        prompt_payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "recent_context": [ev.content[:200] for ev in request.context_evidences[-3:]],
            "graph_context": context_snapshot,
            "coverage_metrics": coverage_snapshot,
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
            parsed = json.loads(response)
            heuristic_metadata = heuristic_note.metadata or {}
            gap_trigger = bool(parsed.get("gap_trigger")) or heuristic_metadata.get("gap_trigger", False)
            missing_topics = self._merge_missing_topics(
                parsed.get("missing_topics"),
                heuristic_metadata.get("missing_topics"),
                coverage_snapshot.get("missing_topics"),
            )
            return ThinkNote(
                plan_step_id=request.plan_step,
                reasoning=str(parsed.get("reasoning") or heuristic_note.reasoning),
                confidence_delta=self._coerce_float(parsed.get("confidence_delta"), heuristic_note.confidence_delta),
                coverage_delta=self._coerce_float(parsed.get("coverage_delta"), heuristic_note.coverage_delta),
                next_actions=self._coerce_actions(parsed.get("next_actions"), heuristic_note.next_actions),
                metadata={
                    "raw": parsed,
                    "heuristic": heuristic_note.model_dump(),
                    "graph_context": context_snapshot,
                    "coverage_metrics": coverage_snapshot,
                    "gap_trigger": gap_trigger,
                    "missing_topics": missing_topics,
                },
            )
        except Exception:
            return heuristic_note

    def _heuristic_note(
        self,
        request: ToolRunRequest,
        coverage_snapshot: Dict[str, Any],
        context_snapshot: Dict[str, Any],
    ) -> ThinkNote:
        context_size = len(request.context_evidences)
        coverage_delta = self._coerce_float(coverage_snapshot.get("coverage_score"), None)
        if coverage_delta is None:
            coverage_delta = min(1.0, 0.1 * context_size)
        confidence_delta = self._coerce_float(coverage_snapshot.get("confidence_score"), None)
        if confidence_delta is None:
            confidence_delta = 0.1 if context_size else -0.2
        next_actions = []
        if not context_size:
            next_actions.append("Trigger fast graph probe to gather seed facts.")
        else:
            next_actions.append("Continue with heavy reasoning step using cached context.")
        gap_trigger = self._should_gap_trigger(coverage_snapshot, context_size)
        return ThinkNote(
            plan_step_id=request.plan_step,
            reasoning="Think window executed heuristically based on available evidences.",
            confidence_delta=confidence_delta,
            coverage_delta=coverage_delta,
            next_actions=next_actions,
            metadata={
                "context_size": context_size,
                "graph_context": context_snapshot,
                "coverage_metrics": coverage_snapshot,
                "gap_trigger": gap_trigger,
                "missing_topics": coverage_snapshot.get("missing_topics") or [],
            },
        )

    @staticmethod
    def _coerce_float(value: Any, fallback: Optional[float]) -> Optional[float]:
        try:
            if value is None:
                return fallback
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _coerce_actions(payload: Any, fallback: List[str]) -> List[str]:
        if isinstance(payload, list):
            actions = [str(item) for item in payload if str(item).strip()]
            if actions:
                return actions
        return fallback

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

    @staticmethod
    def _should_gap_trigger(coverage_snapshot: Dict[str, Any], context_size: int) -> bool:
        coverage_score = coverage_snapshot.get("coverage_score")
        coverage_ratio = coverage_snapshot.get("coverage_ratio")
        missing_topics = coverage_snapshot.get("missing_topics") or []
        if isinstance(coverage_score, (int, float)) and coverage_score < 0.6:
            return True
        if isinstance(coverage_ratio, (int, float)) and coverage_ratio < 0.5:
            return True
        if missing_topics:
            return True
        if context_size == 0:
            return True
        return False
