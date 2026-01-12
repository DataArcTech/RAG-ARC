from typing import Any, Dict, List, Sequence

from core.deepsearch.trace import emit_trace
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.deepsearch import GraphQueryContext


class DeepSearchServiceInitialThinkMixin:
    async def _emit_initial_think(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
        plan_steps: Sequence[Dict[str, Any]],
    ) -> None:
        tool_manager = getattr(self, "tool_manager", None)
        if not tool_manager:
            return

        try:
            total_steps = len(plan_steps) if plan_steps is not None else 0
        except Exception:
            total_steps = 0

        payload = {
            "question": question,
            "plan_step": "think_init",
            "context_evidences": [],
            "adapter": getattr(getattr(self, "graph_loop", None), "adapter", None),
            "access_scope": scope,
            "extra": {"trigger": "initial_think", "plan_steps": list(plan_steps)},
            "graph_context": reasoning_context.model_dump(exclude_none=True),
            "coverage_metrics": {
                "evidence_count": 0,
                "unique_source_count": 0,
                "completed_steps": 0,
                "total_steps": total_steps,
                "coverage_ratio": 0.0,
                "plan_progress_ratio": 0.0,
                "expected_min_chunks": int(self.config["coverage_expected_min_chunks"]),
                "coverage_score": 0.0,
                "confidence_score": None,
                "missing_topics": [],
            },
        }
        try:
            result = await tool_manager.invoke(self._tool_names()["think_tool"], payload=payload)
        except Exception:
            return

        notes = getattr(result, "think_notes", None)
        if not notes:
            return

        lines: List[str] = ["Initial think checkpoint (before execution)."]
        for idx, note in enumerate(notes, start=1):
            lines.append(f"note_{idx}. reasoning={note.reasoning}")
            if note.next_actions:
                lines.append(f"note_{idx}. next_actions={note.next_actions}")
            if note.coverage_delta is not None:
                lines.append(f"note_{idx}. coverage_delta={note.coverage_delta}")
            if note.confidence_delta is not None:
                lines.append(f"note_{idx}. confidence_delta={note.confidence_delta}")
            missing = None
            if isinstance(note.metadata, dict):
                missing = note.metadata.get("missing_topics")
            if isinstance(missing, list) and missing:
                lines.append(f"note_{idx}. missing_topics={missing}")

        await emit_trace("think", "\n".join(lines), meta={"stage": "think_init", "plan_step": "think_init"})

