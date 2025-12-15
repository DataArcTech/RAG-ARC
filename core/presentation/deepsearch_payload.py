"""Utilities for trimming DeepSearch payloads before returning them to clients."""
from copy import deepcopy
from typing import Any, Dict, List, Optional

from config.output_limits import (
    DEEPSEARCH_MAX_EXTERNAL_CALLS,
    DEEPSEARCH_MAX_REASONING_STEPS,
    DEEPSEARCH_MAX_STAGE_HISTORY,
    DEEPSEARCH_MAX_TOOL_METADATA,
    DEEPSEARCH_TOP_CHUNKS,
)
from core.presentation.evidence import build_deepsearch_evidence


def _truncate_evidence_list(block: Optional[Dict[str, Any]], key: str, limit: Optional[int]) -> None:
    if block is None or limit is None:
        return
    items = block.get(key)
    if isinstance(items, list):
        block[key] = items[:limit]


def _head(items: List[Any], limit: int | None) -> List[Any]:
    if limit is None:
        return items
    return items[:max(limit, 0)]


def _tail(items: List[Any], limit: int | None) -> List[Any]:
    if limit is None:
        return items
    if limit <= 0:
        return []
    return items[-limit:]


def trim_deepsearch_payload(
    result: Dict[str, Any],
    *,
    include_evidence: bool = False,
    chunk_limit: Optional[int] = None,
    graph_store: Any | None = None,
) -> Dict[str, Any]:
    """Return a lightweight copy of the DeepSearch result with optional evidence attachment."""

    source = deepcopy(result)
    cap = DEEPSEARCH_TOP_CHUNKS if chunk_limit is None else chunk_limit
    _truncate_evidence_list(source.get("report"), "evidences", cap)
    _truncate_evidence_list(source.get("reasoning"), "evidences", cap)

    evidence_payload = (
        build_deepsearch_evidence(source, chunk_limit=cap, graph_store=graph_store)
        if graph_store is not None
        else build_deepsearch_evidence(source, chunk_limit=cap)
    )
    graph_chain = evidence_payload.get("graph_chain") or source.get("graph_chain") or []
    evidence_block = (
        evidence_payload if include_evidence else _slim_evidence_payload(evidence_payload)
    )

    trimmed_plan = _trim_plan_block(source.get("plan"))
    trimmed_reasoning = _trim_reasoning_block(source.get("reasoning"))
    trimmed_report = _trim_report_block(source.get("report"), cap)
    trimmed_state = _trim_state_block(source.get("state"))

    question = (
        (trimmed_plan.get("question") if isinstance(trimmed_plan, dict) else None)
        or trimmed_report.get("question")
        or trimmed_reasoning.get("question")
    )

    trimmed_payload: Dict[str, Any] = {
        "plan": trimmed_plan,
        "reasoning": trimmed_reasoning,
        "report": trimmed_report,
        "state": trimmed_state,
        "graph_chain": graph_chain,
        "question": question,
        "tool_runs": _summarize_tool_runs(result.get("reasoning")),
        "overview": _build_overview(trimmed_plan, trimmed_reasoning, trimmed_report),
    }
    request_metadata = trimmed_state.get("request_metadata") if isinstance(trimmed_state, dict) else None
    if request_metadata:
        trimmed_payload["request_metadata"] = request_metadata
    trimmed_payload["evidence"] = evidence_block
    return trimmed_payload


def _summarize_reasoning_steps(reasoning_block: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(reasoning_block, dict):
        return []
    steps = reasoning_block.get("reasoning_steps") or []
    if not isinstance(steps, list):
        return []
    summaries: List[Dict[str, Any]] = []
    for step in _head(steps, DEEPSEARCH_MAX_REASONING_STEPS):
        if not isinstance(step, dict):
            continue
        diagnostics = step.get("diagnostics") or {}
        metadata = step.get("metadata") or {}
        tool_name = (
            diagnostics.get("tool")
            or metadata.get("tool")
            or step.get("tool")
        )
        compact_diag = _compact_diagnostics(diagnostics)
        summaries.append(
            {
                "step_id": step.get("step_id"),
                "description": step.get("description"),
                "channel": step.get("channel"),
                "tool": tool_name,
                "status": step.get("status"),
                "output_summary": step.get("output_summary"),
                "evidence_ids": step.get("produced_evidence_ids") or step.get("evidence_ids"),
                "think_notes": step.get("think_notes"),
                "diagnostics": compact_diag or None,
            }
        )
    return summaries


def _compact_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = {"confidence", "coverage", "reason", "latency_ms", "tool"}
    compact = {key: diagnostics.get(key) for key in allowed_keys if diagnostics.get(key) is not None}
    return compact


def _trim_plan_block(plan_block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(plan_block, dict):
        return {}
    plan_payload = plan_block.get("plan")
    if not isinstance(plan_payload, dict):
        return {}
    return {
        "plan_id": plan_payload.get("plan_id"),
        "question": plan_payload.get("question"),
        "mode": plan_payload.get("mode"),
        "steps": plan_payload.get("steps") or [],
    }


def _trim_reasoning_block(reasoning_block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(reasoning_block, dict):
        return {"reasoning_steps": [], "think_notes": []}
    return {
        "question": reasoning_block.get("question"),
        "reasoning_steps": _summarize_reasoning_steps(reasoning_block),
        "coverage_metrics": reasoning_block.get("coverage_metrics") or {},
        "gap_result": reasoning_block.get("gap_result"),
        "think_notes": list(reasoning_block.get("think_notes") or []),
    }


def _trim_report_block(report_block: Optional[Dict[str, Any]], limit: Optional[int]) -> Dict[str, Any]:
    if not isinstance(report_block, dict):
        return {}
    question = report_block.get("question")
    evidences = report_block.get("evidences") or []
    if isinstance(evidences, list) and limit is not None:
        evidences = evidences[:limit]
    trimmed = {
        "question": question,
        "answer": report_block.get("answer"),
        "highlights": report_block.get("highlights") or [],
        "evidences": evidences,
        "metadata": _sanitize_report_metadata(report_block.get("metadata")),
    }
    structured = report_block.get("structured_report")
    if isinstance(structured, dict):
        trimmed["structured_report"] = structured
    return trimmed


def _trim_state_block(state_block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state_block, dict):
        return {}
    telemetry = (state_block.get("cost_telemetry") or {}).get("stage_timings") or {}
    stage_history = list(state_block.get("stage_history") or [])
    payload = {
        "run_id": state_block.get("run_id"),
        "stage": state_block.get("stage"),
        "stage_history": _tail(stage_history, DEEPSEARCH_MAX_STAGE_HISTORY),
        "plan_metadata": state_block.get("plan_metadata") or {},
        "cost_telemetry": {"stage_timings": telemetry},
    }
    external_calls = state_block.get("external_calls") or []
    if external_calls:
        payload["external_calls"] = _head(external_calls, DEEPSEARCH_MAX_EXTERNAL_CALLS)
    if state_block.get("request_metadata"):
        payload["request_metadata"] = state_block["request_metadata"]
    return payload


def _sanitize_report_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None
    allowed_keys = {
        "adapter_metadata",
        "graph_summary",
        "plan",
        "coverage_metrics",
        "pending_external",
        "request_context",
        "report_profile",
    }
    sanitized: Dict[str, Any] = {}
    for key in allowed_keys:
        if metadata.get(key) is not None:
            sanitized[key] = metadata[key]
    tool_results = metadata.get("tool_results") or []
    if tool_results:
        sanitized["tool_results"] = _head(list(tool_results), DEEPSEARCH_MAX_TOOL_METADATA)
    think_notes = metadata.get("think_notes") or []
    if think_notes:
        sanitized["think_notes"] = _head(list(think_notes), DEEPSEARCH_MAX_REASONING_STEPS)
    if not sanitized:
        return None
    return sanitized


def _summarize_tool_runs(reasoning_block: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(reasoning_block, dict):
        return []
    runs = reasoning_block.get("tool_results") or []
    summaries: List[Dict[str, Any]] = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        result_payload = entry.get("result") or {}
        if isinstance(result_payload, dict):
            summary_text = result_payload.get("summary")
            diagnostics = result_payload.get("diagnostics") or {}
            think_notes = result_payload.get("think_notes") or []
        else:
            summary_text = None
            diagnostics = {}
            think_notes = []
        summaries.append(
            {
                "plan_step_id": entry.get("plan_step_id"),
                "tool_name": entry.get("tool_name"),
                "channel": entry.get("channel"),
                "summary": summary_text,
                "diagnostics": diagnostics or None,
                "think_notes": think_notes,
            }
        )
    return summaries


__all__ = ["trim_deepsearch_payload"]


def _slim_evidence_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunks": payload.get("chunks") or [],
        "seed_entities": payload.get("seed_entities") or [],
    }


def _build_overview(plan_block: Dict[str, Any], reasoning_block: Dict[str, Any], report_block: Dict[str, Any]) -> Dict[str, Any]:
    plan_steps = plan_block.get("steps") if isinstance(plan_block, dict) else []
    reasoning_steps = reasoning_block.get("reasoning_steps") if isinstance(reasoning_block, dict) else []
    evidences = report_block.get("evidences") if isinstance(report_block, dict) else []
    return {
        "plan_step_count": len(plan_steps or []),
        "reasoning_step_count": len(reasoning_steps or []),
        "has_think_notes": bool(reasoning_block.get("think_notes")) if isinstance(reasoning_block, dict) else False,
        "evidence_count": len(evidences or []),
    }
