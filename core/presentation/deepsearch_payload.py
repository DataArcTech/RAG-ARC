"""Utilities for trimming DeepSearch payloads before returning them to clients."""
from copy import deepcopy
from typing import Any, Dict, List, Optional

from config.output_limits import DEEPSEARCH_TOP_CHUNKS
from core.presentation.evidence import build_deepsearch_evidence

_MAX_REASONING_STEPS = 32


def _truncate_evidence_list(block: Optional[Dict[str, Any]], key: str, limit: Optional[int]) -> None:
    if block is None or limit is None:
        return
    items = block.get(key)
    if isinstance(items, list):
        block[key] = items[:limit]


def trim_deepsearch_payload(
    result: Dict[str, Any],
    *,
    include_evidence: bool = False,
    chunk_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a lightweight copy of the DeepSearch result with optional evidence attachment."""

    source = deepcopy(result)
    cap = DEEPSEARCH_TOP_CHUNKS if chunk_limit is None else chunk_limit
    _truncate_evidence_list(source.get("report"), "evidences", cap)
    _truncate_evidence_list(source.get("reasoning"), "evidences", cap)

    evidence_payload = build_deepsearch_evidence(source, chunk_limit=cap)
    graph_chain = evidence_payload.get("graph_chain") or source.get("graph_chain") or []

    trimmed_plan = _trim_plan_block(source.get("plan"))
    trimmed_reasoning = _trim_reasoning_block(source.get("reasoning"))
    trimmed_report = _trim_report_block(source.get("report"), cap)
    trimmed_state = _trim_state_block(source.get("state"))

    trimmed_payload: Dict[str, Any] = {
        "plan": trimmed_plan,
        "reasoning": trimmed_reasoning,
        "report": trimmed_report,
        "state": trimmed_state,
        "graph_chain": graph_chain,
        "reasoning_steps": trimmed_reasoning.get("reasoning_steps") or [],
    }
    if include_evidence:
        trimmed_payload["evidence"] = evidence_payload
    return trimmed_payload


def _summarize_reasoning_steps(reasoning_block: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(reasoning_block, dict):
        return []
    steps = reasoning_block.get("reasoning_steps") or []
    if not isinstance(steps, list):
        return []
    summaries: List[Dict[str, Any]] = []
    for step in steps[:_MAX_REASONING_STEPS]:
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
        return {"reasoning_steps": []}
    return {
        "question": reasoning_block.get("question"),
        "reasoning_steps": _summarize_reasoning_steps(reasoning_block),
        "coverage_metrics": reasoning_block.get("coverage_metrics") or {},
        "gap_result": reasoning_block.get("gap_result"),
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
        "metadata": None,
    }
    structured = report_block.get("structured_report")
    if isinstance(structured, dict):
        trimmed["structured_report"] = structured
    return trimmed


def _trim_state_block(state_block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state_block, dict):
        return {}
    telemetry = (state_block.get("cost_telemetry") or {}).get("stage_timings") or {}
    return {"cost_telemetry": {"stage_timings": telemetry}}


__all__ = ["trim_deepsearch_payload"]
