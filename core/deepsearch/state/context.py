"""Runtime state tracker for the DeepSearch pipeline."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DeepSearchState:
    """Track planner output, reasoning traces, gap decisions, and reports for one run."""

    config_fingerprint: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stage: str = field(default="created")
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    stage_listener: Optional[Callable[[Dict[str, Any], "DeepSearchState"], None]] = field(
        default=None, repr=False, compare=False
    )
    plan_metadata: Dict[str, Any] = field(default_factory=dict)
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: Dict[str, Any] = field(default_factory=dict)
    gap_result: Optional[Dict[str, Any]] = None
    external_calls: List[Dict[str, Any]] = field(default_factory=list)
    cost_telemetry: Dict[str, Any] = field(default_factory=dict)
    report_payload: Optional[Dict[str, Any]] = None
    quality_gates: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    request_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record = {"stage": self.stage, "timestamp": _utc_now()}
        self.stage_history.append(record)
        self._emit_stage(record)

    # ------------------------------------------------------------------
    def record_plan(self, plan_result: Dict[str, Any]) -> None:
        plan_payload = plan_result.get("plan") or {}
        steps = plan_payload.get("steps") or plan_result.get("steps") or []
        plan_id = plan_result.get("plan_id") or plan_payload.get("plan_id")
        self.plan_steps = steps
        self.plan_metadata = {
            "plan_id": plan_id,
            "artifact_path": plan_result.get("artifact_path"),
            "created_at": plan_payload.get("created_at") or plan_result.get("created_at"),
            "mode": plan_payload.get("mode") or plan_result.get("mode"),
        }
        self.transition_stage(
            "planned",
            metadata={
                "plan_id": plan_id,
                "step_count": len(steps),
            },
        )

    def record_reasoning(self, trace: Dict[str, Any]) -> None:
        self.reasoning_trace = trace or {}
        completed = sum(1 for step in self.reasoning_trace.get("reasoning_steps", []) if step.get("status") == "done")
        self.transition_stage(
            "reasoned",
            metadata={
                "evidence_count": len(self.reasoning_trace.get("evidences", []) or []),
                "completed_steps": completed,
            },
        )

    def record_gap_result(self, gap_payload: Dict[str, Any]) -> None:
        self.gap_result = gap_payload or {}
        if self.reasoning_trace is not None:
            self.reasoning_trace["gap_result"] = self.gap_result
        self.transition_stage(
            "gap_evaluated",
            metadata={
                "should_trigger_external": bool(self.gap_result.get("should_trigger_external")),
                "reason": self.gap_result.get("reason"),
            },
        )

    def record_external_call(self, call_log: Dict[str, Any]) -> None:
        if not call_log:
            return
        self.external_calls.append(call_log)
        self.transition_stage(
            "external_invoked",
            metadata={"total_calls": len(self.external_calls)},
        )

    def extend_external_calls(self, call_logs: List[Dict[str, Any]]) -> None:
        for log in call_logs:
            self.record_external_call(log)

    def record_cost(self, label: str, payload: Dict[str, Any]) -> None:
        if not label:
            return
        bucket = self.cost_telemetry.setdefault(label, {})
        bucket.update(payload or {})

    def record_report(self, report_payload: Dict[str, Any]) -> None:
        self.report_payload = report_payload or {}
        self.transition_stage(
            "reported",
            metadata={
                "answer_length": len((self.report_payload.get("answer") or "")),
                "evidence_count": len(self.report_payload.get("evidences") or []),
            },
        )

    def record_quality_gate(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        self.quality_gates.append(payload)
        record = {"stage": "quality_gated", "timestamp": _utc_now()}
        record["metadata"] = {
            "round": payload.get("round"),
            "passed": payload.get("passed"),
            "should_iterate": payload.get("should_iterate"),
            "enabled": payload.get("enabled"),
        }
        self.stage_history.append(record)
        self._emit_stage(record)

    def mark_failed(self, reason: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        entry = {"reason": reason, "timestamp": _utc_now()}
        if details:
            entry["details"] = details
        self.errors.append(entry)
        self.transition_stage("failed", metadata=entry)

    def append_error(self, message: str, *, stage: Optional[str] = None) -> None:
        entry = {"message": message, "timestamp": _utc_now()}
        if stage:
            entry["stage"] = stage
        self.errors.append(entry)

    def record_request_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        if not metadata:
            return
        self.request_metadata = dict(metadata)

    def transition_stage(self, stage: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        normalized = (stage or "").strip().lower() or "unknown"
        self.stage = normalized
        record = {"stage": normalized, "timestamp": _utc_now()}
        if metadata:
            record["metadata"] = metadata
        self.stage_history.append(record)
        self._emit_stage(record)

    def _emit_stage(self, record: Dict[str, Any]) -> None:
        listener = self.stage_listener
        if not callable(listener):
            return
        try:
            listener(record, self)
        except Exception:
            # Progress callbacks must never break the main pipeline
            return

    def snapshot(self) -> Dict[str, Any]:
        snapshot = {
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "stage": self.stage,
            "stage_history": list(self.stage_history),
            "plan_metadata": self.plan_metadata,
            "plan_steps": self.plan_steps,
            "reasoning_trace": self.reasoning_trace,
            "gap_result": self.gap_result,
            "external_calls": self.external_calls,
            "cost_telemetry": self.cost_telemetry,
            "report": self.report_payload,
            "quality_gates": list(self.quality_gates),
            "errors": self.errors,
            "request_metadata": self.request_metadata,
        }
        snapshot["kpis"] = self._aggregate_kpis()
        snapshot["error_summary"] = self._aggregate_error_summary()
        return snapshot

    # ------------------------------------------------------------------
    def _aggregate_kpis(self) -> Dict[str, Any]:
        trace = self.reasoning_trace or {}
        evidences = trace.get("evidences") or []
        if not isinstance(evidences, list):
            evidences = []
        sources = []
        for item in evidences:
            if isinstance(item, dict) and item.get("source"):
                sources.append(str(item.get("source")))
        tool_invocations = 0
        tool_failures = 0
        tool_timeouts = 0
        reasoning_steps = trace.get("reasoning_steps") or []
        if isinstance(reasoning_steps, list):
            for step in reasoning_steps:
                if not isinstance(step, dict):
                    continue
                logs = step.get("tool_logs") or []
                if isinstance(logs, list):
                    tool_invocations += len(logs)
                if step.get("status") == "failed":
                    tool_failures += 1
                diagnostics = step.get("diagnostics") or {}
                if isinstance(diagnostics, dict) and diagnostics.get("reason") == "tool_timeout":
                    tool_timeouts += 1

        remote_fallbacks = 0
        tool_results = trace.get("tool_results") or []
        if isinstance(tool_results, list):
            for entry in tool_results:
                if not isinstance(entry, dict):
                    continue
                result = entry.get("result") or {}
                diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
                if isinstance(diagnostics, dict) and diagnostics.get("remote_fallback_reason"):
                    remote_fallbacks += 1

        coverage = trace.get("coverage_metrics") or {}
        worker_errors = 0
        if isinstance(coverage, dict) and isinstance(coverage.get("worker_error_count"), int):
            worker_errors = int(coverage.get("worker_error_count") or 0)

        return {
            "evidence_count": len(evidences),
            "unique_source_count": len({s for s in sources if s}),
            "tool_invocation_count": tool_invocations,
            "tool_failure_steps": tool_failures,
            "tool_timeout_steps": tool_timeouts,
            "remote_fallback_count": remote_fallbacks,
            "external_call_count": len(self.external_calls or []),
            "worker_error_count": worker_errors,
        }

    def _aggregate_error_summary(self) -> Dict[str, Any]:
        reasons: Dict[str, int] = {}
        for entry in self.errors or []:
            if not isinstance(entry, dict):
                continue
            reason = entry.get("reason") or entry.get("message") or "unknown"
            token = str(reason).strip() or "unknown"
            reasons[token] = reasons.get(token, 0) + 1

        trace = self.reasoning_trace or {}
        coverage = trace.get("coverage_metrics") or {}
        worker_errors = coverage.get("worker_errors") if isinstance(coverage, dict) else None
        worker_error_count = 0
        if isinstance(worker_errors, list):
            worker_error_count = len([e for e in worker_errors if isinstance(e, dict) and e.get("error")])

        return {
            "error_count": len(self.errors or []),
            "error_reasons": reasons,
            "worker_error_count": worker_error_count,
        }
