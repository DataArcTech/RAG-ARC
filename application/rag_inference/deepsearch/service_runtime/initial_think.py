import asyncio
from typing import Any, Dict, List, Sequence

from core.deepsearch.trace import emit_trace
from core.deepsearch.memory.plan_state import PlanState, update_plan_from_think_notes
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.deepsearch import GraphQueryContext


class DeepSearchServiceInitialThinkMixin:
    def _resolve_tool_timeout_seconds(self) -> float | None:
        cfg = None
        try:
            cfg = (getattr(self, "config", None) or {}).get("graph_reasoning")
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            raw = cfg.get("tool_timeout_seconds")
        else:
            raw = None
        if raw is None:
            return None
        value = float(raw)
        if value < 0:
            raise ValueError("graph_reasoning.tool_timeout_seconds must be >= 0")
        return value

    def _think_tool_name(self) -> str:
        if not isinstance(self.config, dict):
            raise ValueError("DeepSearchService config must be a dict")
        tool_name = str(self.config.get("think_tool") or "").strip()
        if not tool_name:
            raise ValueError("DeepSearchService config.think_tool is required")
        return tool_name

    async def _run_initial_think(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
        plan_steps: Sequence[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        tool_manager = getattr(self, "tool_manager", None)
        if not tool_manager:
            raise RuntimeError("DeepSearchService requires a tool_manager for initial_think")

        steps = list(plan_steps or [])
        total_steps = len(steps)

        plan_state = PlanState()
        if isinstance(reasoning_context.metadata, dict):
            plan_state.update(reasoning_context.metadata.get("runtime_plan"))
        payload = {
            "question": question,
            "plan_step": "think_init",
            "context_evidences": [],
            "adapter": getattr(getattr(self, "graph_loop", None), "adapter", None),
            "access_scope": scope,
            "extra": {
                "trigger": "initial_think",
                "plan_steps": steps,
                "current_plan": list(plan_state.items),
                "think_mode": "initial",
            },
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
        timeout = self._resolve_tool_timeout_seconds()
        invocation = tool_manager.invoke(self._think_tool_name(), payload=payload)
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(invocation, timeout=timeout)
        else:
            result = await invocation

        notes = getattr(result, "think_notes", None)
        if not notes:
            raise RuntimeError("Initial think returned no think_notes")

        raw = None
        for note in reversed(list(notes)):
            raw = note.metadata.get("raw") if isinstance(note.metadata, dict) else None
            if isinstance(raw, dict):
                break
        if not isinstance(raw, dict):
            raise RuntimeError("Initial think returned no structured payload")
        report_needed = raw.get("report_needed")
        if report_needed is None:
            raise RuntimeError("Initial think missing report_needed")
        report_style_raw = raw.get("report_style")
        report_style = str(report_style_raw or "").strip().lower() if report_style_raw is not None else ""
        if report_style not in {"deepsearch", "research"}:
            report_style = "deepsearch"
        if isinstance(reasoning_context.metadata, dict):
            reasoning_context.metadata["report_style"] = report_style

        if update_plan_from_think_notes(plan_state, think_notes=notes):
            reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
            await emit_trace(
                "write_outline",
                plan_state.markdown,
                meta={
                    "stage": "think_init",
                    "plan_step_id": "think_init",
                    "plan_version": plan_state.version,
                    "plan_items": list(plan_state.items),
                },
            )

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
        note_payloads = [note.model_dump(exclude_none=True) for note in notes]
        return {
            "report_needed": bool(report_needed),
            "report_style": report_style,
            "plan_state": plan_state,
            "think_notes": note_payloads,
            "think_notes_obj": list(notes),
            "raw": raw,
        }

    async def _run_final_think(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
        evidences: Sequence[Dict[str, Any]] | None,
        coverage_metrics: Dict[str, Any] | None,
        plan_items: Sequence[Dict[str, Any]] | None,
        report_needed: bool | None = None,
        final_answer_mode: str | None = None,
    ) -> Dict[str, Any]:
        tool_manager = getattr(self, "tool_manager", None)
        if not tool_manager:
            raise RuntimeError("DeepSearchService requires a tool_manager for final_think")

        plan_state = PlanState()
        plan_state.update(plan_items or [])
        extra = {
            "trigger": "final_think",
            "current_plan": list(plan_state.items),
            "think_mode": "final",
        }
        if report_needed is not None:
            extra["report_needed"] = bool(report_needed)
        if final_answer_mode:
            extra["final_answer_mode"] = str(final_answer_mode)
        payload = {
            "question": question,
            "plan_step": "think_final",
            "context_evidences": list(evidences or []),
            "adapter": getattr(getattr(self, "graph_loop", None), "adapter", None),
            "access_scope": scope,
            "extra": extra,
            "graph_context": reasoning_context.model_dump(exclude_none=True),
            "coverage_metrics": coverage_metrics or {},
        }
        timeout = self._resolve_tool_timeout_seconds()
        invocation = tool_manager.invoke(self._think_tool_name(), payload=payload)
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(invocation, timeout=timeout)
        else:
            result = await invocation

        notes = getattr(result, "think_notes", None) or []
        raw = None
        for note in reversed(list(notes)):
            raw = note.metadata.get("raw") if isinstance(note.metadata, dict) else None
            if isinstance(raw, dict):
                break
        raw = raw if isinstance(raw, dict) else {}

        if notes and update_plan_from_think_notes(plan_state, think_notes=notes):
            await emit_trace(
                "write_outline",
                plan_state.markdown,
                meta={
                    "stage": "final_think",
                    "plan_step_id": "think_final",
                    "plan_version": plan_state.version,
                    "plan_items": list(plan_state.items),
                },
            )

        lines: List[str] = ["Final think checkpoint."]
        for idx, note in enumerate(notes, start=1):
            lines.append(f"note_{idx}. reasoning={note.reasoning}")
        await emit_trace("think", "\n".join(lines), meta={"stage": "final_think", "plan_step": "think_final"})
        note_payloads = [note.model_dump(exclude_none=True) for note in notes]
        return {
            "think_notes": note_payloads,
            "plan_state": plan_state,
            "raw": raw,
        }
