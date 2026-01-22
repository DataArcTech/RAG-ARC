import logging
from typing import Any, Dict, List, Optional, Sequence

from core.deepsearch.report import DeepSearchQualityGate, QualityGateConfig
from encapsulation.data_model.deepsearch import ThinkNote
from core.deepsearch.state import DeepSearchState

logger = logging.getLogger(__name__)


class DeepSearchServiceQualityMixin:
    def _resolve_quality_gate_config(self) -> QualityGateConfig:
        if not isinstance(self.config, dict) or not isinstance(self.config.get("quality_loop"), dict):
            raise ValueError("DeepSearchService missing required config.quality_loop")
        return QualityGateConfig.model_validate(self.config["quality_loop"])

    async def _quality_gate_stage(
        self,
        *,
        gate: DeepSearchQualityGate,
        question: str,
        structured_report: Optional[Dict[str, Any]],
        evidences: Sequence[Dict[str, Any]],
        round_idx: int,
    ) -> Dict[str, Any]:
        result = await gate.evaluate(
            question=question,
            structured_report=structured_report,
            evidences=evidences,
        )
        payload = result.model_dump()
        payload["round"] = round_idx
        return payload

    @staticmethod
    def _build_followup_think_notes(*, actions: Sequence[Dict[str, Any]], round_idx: int) -> List[ThinkNote]:
        tool_calls: List[Dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action") != "graph_search":
                continue
            query = str(action.get("query") or "").strip()
            if not query:
                continue
            tool_calls.append(
                {
                    "tool_name": "explore",
                    "tool_args": {"actions": [{"tool": "search", "args": {"focus_query": query}}]},
                    "rationale": f"Quality gate follow-up: {query}",
                    "parallelizable": True,
                }
            )

        if not tool_calls:
            return []
        return [
            ThinkNote(
                plan_step_id=f"quality_followup_r{round_idx + 1}",
                reasoning="Quality gate follow-up actions.",
                metadata={"raw": {"tool_calls": tool_calls}},
            )
        ]

    @staticmethod
    def _merge_reasoning_traces(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(base or {})
        for key in ("graph_traversals", "reasoning_steps", "tool_results", "think_notes"):
            merged_list = list(merged.get(key) or [])
            add_list = list(incoming.get(key) or [])
            merged[key] = merged_list + add_list

        def _merge_by_chunk_id(items: Sequence[Any]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                out.append(item)
            return out

        merged["evidences"] = _merge_by_chunk_id(list(merged.get("evidences") or []) + list(incoming.get("evidences") or []))

        plan_steps: List[Dict[str, Any]] = []
        seen_steps: set[str] = set()
        for item in list(merged.get("plan_steps") or []) + list(incoming.get("plan_steps") or []):
            if not isinstance(item, dict):
                continue
            step_id = str(item.get("step_id") or "").strip()
            if not step_id or step_id in seen_steps:
                continue
            seen_steps.add(step_id)
            plan_steps.append(item)
        merged["plan_steps"] = plan_steps

        base_cov = merged.get("coverage_metrics") if isinstance(merged.get("coverage_metrics"), dict) else {}
        inc_cov = incoming.get("coverage_metrics") if isinstance(incoming.get("coverage_metrics"), dict) else {}
        cov = dict(base_cov or {})
        cov.update(inc_cov or {})
        cov.setdefault("evidence_count", len(merged.get("evidences") or []))
        merged["coverage_metrics"] = cov
        if "runtime_plan" in incoming:
            merged["runtime_plan"] = incoming.get("runtime_plan")
        return merged

    @staticmethod
    def _surface_worker_failures(state: DeepSearchState, reasoning_trace: Dict[str, Any]) -> None:
        if not isinstance(reasoning_trace, dict):
            return
        coverage = reasoning_trace.get("coverage_metrics") or {}
        if not isinstance(coverage, dict):
            return
        errors = coverage.get("worker_errors") or []
        if isinstance(errors, list) and errors:
            count = coverage.get("worker_error_count")
            try:
                count_int = int(count) if count is not None else len(errors)
            except (TypeError, ValueError):
                count_int = len(errors)
            previews: list[str] = []
            for entry in errors:
                if len(previews) >= 3:
                    break
                if not isinstance(entry, dict):
                    continue
                agent_id = entry.get("agent_id") or "worker"
                code = entry.get("error") or "error"
                previews.append(f"{agent_id}={code}")
            preview = ", ".join(previews)
            message = f"{count_int} worker agent(s) failed"
            if preview:
                message = f"{message}: {preview}"
            state.append_error(message, stage="graph_reasoning")

        probe_errors = coverage.get("probe_errors") or []
        if not isinstance(probe_errors, list) or not probe_errors:
            return
        previews = []
        for entry in probe_errors:
            if len(previews) >= 3:
                break
            if not isinstance(entry, dict):
                continue
            tool = entry.get("tool_name") or "probe"
            code = entry.get("error") or entry.get("error_type") or "error"
            previews.append(f"{tool}={code}")
        preview = ", ".join(previews)
        message = f"{len(probe_errors)} probe tool(s) failed"
        if preview:
            message = f"{message}: {preview}"
        state.append_error(message, stage="graph_reasoning", details={"probe_errors": probe_errors})
