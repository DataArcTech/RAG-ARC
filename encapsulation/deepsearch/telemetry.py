"""Telemetry utilities shared by DeepSearch service/tooling."""
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


class LoggingTelemetryClient:
    """Minimal telemetry client that emits structured logs for DeepSearch tool usage."""

    def log_tool_invocation(self, *, tool_name: str, payload: Dict[str, Any]) -> None:
        evidence_count = payload.get("evidence_count")
        logger.info(
            "deepsearch.tool",
            extra={
                "event": "tool",
                "run_id": payload.get("run_id"),
                "tool_name": tool_name,
                "tool_namespace": payload.get("tool_namespace"),
                "plan_step": payload.get("plan_step"),
                "latency_ms": payload.get("latency_ms"),
                "evidence_count": evidence_count if evidence_count is not None else payload.get("evidences_count"),
                "external_allowed": payload.get("external_allowed"),
                "scope_override_allowed": payload.get("scope_override_allowed"),
                "scope_override_policy": payload.get("scope_override_policy"),
                "mcp_server": payload.get("mcp_server"),
            },
        )

    def log_remote_tool(self, *, tool_name: str, log) -> None:  # noqa: ANN001
        logger.info(
            "deepsearch.tool_remote",
            extra={
                "event": "tool_remote",
                "run_id": (log.extra or {}).get("run_id"),
                "tool_name": tool_name,
                "tool_namespace": (log.extra or {}).get("tool_namespace"),
                "server_name": getattr(log, "server_name", None),
                "latency_ms": getattr(log, "latency_ms", None),
                "evidence_count": (log.extra or {}).get("evidence_count"),
                "external_allowed": (log.extra or {}).get("external_allowed"),
                "scope_override_allowed": (log.extra or {}).get("scope_override_allowed"),
                "scope_override_policy": (log.extra or {}).get("scope_override_policy"),
                "transport": (log.extra or {}).get("transport"),
            },
        )

    def log_gap_detection(self, *, result: Dict[str, Any], context: Dict[str, Any] | None = None) -> None:
        diagnostics = (result or {}).get("diagnostics") or {}
        logger.info(
            "deepsearch.gap",
            extra={
                "event": "gap",
                "run_id": (context or {}).get("run_id"),
                "question": (context or {}).get("question"),
                "external_allowed": diagnostics.get("external_allowed"),
                "should_trigger_external": result.get("should_trigger_external"),
                "evidence_count": diagnostics.get("evidence_count"),
                "coverage_score": result.get("coverage_score"),
                "confidence_score": result.get("confidence_score"),
                "missing_topics_count": len(result.get("missing_topics") or []),
            },
        )

    def log_external_channel(self, *, payload: Dict[str, Any]) -> None:
        logger.info(
            "deepsearch.external",
            extra={
                "event": "external",
                "run_id": payload.get("run_id"),
                "provider": payload.get("provider"),
                "step_id": payload.get("step_id"),
                "status": payload.get("status"),
                "latency_ms": payload.get("latency_ms"),
                "evidence_count": payload.get("evidence_count"),
            },
        )

