import logging
from typing import Any, Dict, List, Optional, Sequence

from core.deepsearch.report import DeepSearchQualityGate, QualityGateConfig
from core.deepsearch.state import DeepSearchState

logger = logging.getLogger(__name__)


class DeepSearchServiceQualityMixin:
    def _resolve_quality_gate_config(self) -> QualityGateConfig:
        if not isinstance(self.config, dict) or not isinstance(self.config.get("quality_loop"), dict):
            raise ValueError("DeepSearchService missing required config.quality_loop")
        return QualityGateConfig.model_validate(self.config["quality_loop"])

    def _tool_names(self) -> Dict[str, str]:
        if not isinstance(self.config, dict):
            raise ValueError("DeepSearchService config must be a dict")
        names = self.config.get("tool_names")
        if not isinstance(names, dict):
            raise ValueError("DeepSearchService config.tool_names is required")
        required = ("graph_channel_tool", "web_channel_tool", "text_channel_tool", "think_tool")
        missing = [key for key in required if not str(names.get(key) or "").strip()]
        if missing:
            raise ValueError(f"DeepSearchService config.tool_names missing: {missing}")
        return {k: str(names[k]).strip() for k in required}

    async def _quality_gate_stage(
        self,
        *,
        gate: DeepSearchQualityGate,
        question: str,
        structured_report: Optional[Dict[str, Any]],
        evidences: Sequence[Dict[str, Any]],
        gap_result: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> Dict[str, Any]:
        result = await gate.evaluate(
            question=question,
            structured_report=structured_report,
            evidences=evidences,
            gap_result=gap_result,
            external_allowed=self._external_allowed_flag(),
        )
        payload = result.model_dump()
        payload["round"] = round_idx
        return payload

    def _build_followup_plan_steps(self, *, actions: Sequence[Dict[str, Any]], round_idx: int) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        counter = 0
        graph_tool = self._tool_names()["graph_channel_tool"]
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action") != "graph_search":
                continue
            query = str(action.get("query") or "").strip()
            if not query:
                continue
            counter += 1
            step_id = f"quality_graph_r{round_idx + 1}_{counter:02d}"
            steps.append(
                {
                    "step_id": step_id,
                    "description": f"Quality follow-up graph search: {query}",
                    "channel": "graph",
                    "metadata": {"source": "quality_gate", "round": round_idx + 1},
                    "tool": graph_tool,
                    "tool_args": {"query": query},
                    "requires_external": False,
                    "enabled": True,
                }
            )
        return steps

    def _build_external_tasks_from_actions(
        self,
        *,
        actions: Sequence[Dict[str, Any]],
        round_idx: int,
        question: str,
    ) -> List[Dict[str, Any]]:
        if not actions:
            return []
        web_tool = self._tool_names()["web_channel_tool"]
        tasks: List[Dict[str, Any]] = []
        external_channel = getattr(self, "external_channel", None)
        for idx, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                continue
            if action.get("action") != "external_search":
                continue
            query = str(action.get("query") or "").strip()
            if not query:
                continue
            step_id = f"quality_web_r{round_idx + 1}_{idx:02d}"
            tasks.append(
                {
                    "step_id": step_id,
                    "description": str(action.get("rationale") or "Quality follow-up external search"),
                    "channel": "web",
                    "tool": web_tool,
                    "metadata": {
                        "provider": getattr(external_channel, "default_provider", None) if external_channel else None,
                        "query": query,
                        "source": "quality_gate",
                        "round": round_idx + 1,
                        "question": question,
                    },
                    "tool_args": {"query": query, "reason": "quality_gate"},
                    "requires_external": True,
                }
            )
        return tasks

    @staticmethod
    def _merge_reasoning_traces(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(base or {})
        for key in ("graph_traversals", "reasoning_steps", "tool_results", "think_notes", "pending_external"):
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
        return merged

    @staticmethod
    def _surface_worker_failures(state: DeepSearchState, reasoning_trace: Dict[str, Any]) -> None:
        if not isinstance(reasoning_trace, dict):
            return
        coverage = reasoning_trace.get("coverage_metrics") or {}
        if not isinstance(coverage, dict):
            return
        errors = coverage.get("worker_errors") or []
        if not isinstance(errors, list) or not errors:
            return
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
