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
                "scope_override_allowed": (log.extra or {}).get("scope_override_allowed"),
                "scope_override_policy": (log.extra or {}).get("scope_override_policy"),
                "transport": (log.extra or {}).get("transport"),
            },
        )
