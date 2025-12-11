"""Runtime state tracker for the DeepSearch pipeline."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DeepSearchState:
    """Track planner output, reasoning traces, gap decisions, and reports for one run."""

    config_fingerprint: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stage: str = field(default="created")
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    plan_metadata: Dict[str, Any] = field(default_factory=dict)
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: Dict[str, Any] = field(default_factory=dict)
    gap_result: Optional[Dict[str, Any]] = None
    external_calls: List[Dict[str, Any]] = field(default_factory=list)
    cost_telemetry: Dict[str, Any] = field(default_factory=dict)
    report_payload: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage_history.append({"stage": self.stage, "timestamp": _utc_now()})

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

    def transition_stage(self, stage: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        normalized = (stage or "").strip().lower() or "unknown"
        self.stage = normalized
        record = {"stage": normalized, "timestamp": _utc_now()}
        if metadata:
            record["metadata"] = metadata
        self.stage_history.append(record)

    def snapshot(self) -> Dict[str, Any]:
        return {
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
            "errors": self.errors,
        }
