"""Plan sub-agent primitives for DeepSearch."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphTraversalRecord, ReasoningStepRecord


@dataclass
class SubAgentOutcome:
    """Result payload emitted by each plan sub-agent."""

    step_index: int
    reasoning: ReasoningStepRecord
    traversal: Optional[GraphTraversalRecord]
    evidences: List[EvidenceChunk]
    tool_runs: List[Dict[str, Any]]
    think_notes: List[Dict[str, Any]]


class PlanSubAgent:
    """Lightweight wrapper that delegates execution back to the main orchestrator."""

    def __init__(self, owner: "GraphReasoningLoop", step_index: int, entry: Dict[str, Any], question: str, context):
        from .graph_loop import GraphReasoningLoop  # Local import to avoid cycles

        if not isinstance(owner, GraphReasoningLoop):  # pragma: no cover - defensive guard
            raise TypeError("owner must be a GraphReasoningLoop instance")
        self._owner = owner
        self.step_index = step_index
        self._entry = entry
        self._context = context
        self._question = question

    async def run(self) -> SubAgentOutcome:
        return await self._owner._execute_plan_entry(  # type: ignore[attr-defined]
            step_index=self.step_index,
            entry=self._entry,
            question=self._question,
            context=self._context,
        )
