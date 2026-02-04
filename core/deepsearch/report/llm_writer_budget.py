import json
from typing import Any, Dict, List

from config.core.deepsearch import report_writer_defaults as report_defaults


def dump_json(payload: Any) -> str:
    """Token-efficient JSON rendering for prompts (no pretty indent)."""

    def _default(value: Any) -> str:
        return str(value)

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_default)


def slim_methodology(method: Dict[str, Any], *, level: int) -> Dict[str, Any]:
    """Shrink methodology payload for prompt budgeting."""

    if not isinstance(method, dict):
        return {}
    normalized_level = max(0, int(level))

    final_answer = method.get("final_answer")
    plan_steps = method.get("plan_steps") if isinstance(method.get("plan_steps"), list) else []
    reasoning_steps = method.get("reasoning_steps") if isinstance(method.get("reasoning_steps"), list) else []

    if normalized_level <= 0:
        return method

    if normalized_level == 1:
        slim_plan: list[dict[str, Any]] = []
        for step in plan_steps[: report_defaults.SLIM_METHOD_LEVEL_1_PLAN_STEPS]:
            if isinstance(step, dict):
                slim_plan.append(
                    {
                        "step_id": step.get("step_id"),
                        "description": step.get("description"),
                        "channel": step.get("channel"),
                        "tool": step.get("tool"),
                    }
                )
        slim_reasoning: list[dict[str, Any]] = []
        for step in reasoning_steps[: report_defaults.SLIM_METHOD_LEVEL_1_REASONING_STEPS]:
            if isinstance(step, dict):
                slim_reasoning.append(
                    {
                        "step_id": step.get("step_id"),
                        "description": step.get("description"),
                        "status": step.get("status"),
                        "output_summary": step.get("output_summary"),
                        "tool": step.get("tool"),
                    }
                )
        payload: Dict[str, Any] = {"plan_steps": slim_plan, "reasoning_steps": slim_reasoning}
        if final_answer:
            payload["final_answer"] = final_answer
        return payload

    if normalized_level == 2:
        slim_plan = []
        for step in plan_steps[: report_defaults.SLIM_METHOD_LEVEL_2_PLAN_STEPS]:
            if isinstance(step, dict):
                slim_plan.append({"step_id": step.get("step_id"), "description": step.get("description")})
        slim_reasoning = []
        for step in reasoning_steps[: report_defaults.SLIM_METHOD_LEVEL_2_REASONING_STEPS]:
            if isinstance(step, dict):
                slim_reasoning.append(
                    {"step_id": step.get("step_id"), "output_summary": step.get("output_summary"), "tool": step.get("tool")}
                )
        payload = {"plan_steps": slim_plan, "reasoning_steps": slim_reasoning}
        if final_answer:
            payload["final_answer"] = final_answer
        return payload

    payload = {}
    if final_answer:
        payload["final_answer"] = final_answer
    return payload


def slim_graph_evidence(graph_evidence: Any, *, level: int) -> Dict[str, Any]:
    if not isinstance(graph_evidence, dict):
        return {}
    normalized_level = max(0, int(level))
    if normalized_level <= 0:
        return graph_evidence
    seed_entities = graph_evidence.get("seed_entities") if isinstance(graph_evidence.get("seed_entities"), list) else []
    graph_stats = graph_evidence.get("graph_stats") if isinstance(graph_evidence.get("graph_stats"), dict) else {}
    if normalized_level == 1:
        return {
            "seed_entities": seed_entities[: report_defaults.SLIM_GRAPH_EVIDENCE_LEVEL_SEED_ENTITIES],
            "graph_stats": graph_stats,
        }
    return {"seed_entities": seed_entities[: report_defaults.SLIM_GRAPH_EVIDENCE_LEVEL_SEED_ENTITIES]}


def slim_coverage(coverage: Any, *, level: int) -> Dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    normalized_level = max(0, int(level))

    if normalized_level <= 0:
        out = dict(coverage)
        return {
            "evidence_count": out.get("evidence_count"),
            "unique_source_count": out.get("unique_source_count"),
            "completed_steps": out.get("completed_steps"),
            "total_steps": out.get("total_steps"),
            "coverage_ratio": out.get("coverage_ratio"),
            "plan_progress_ratio": out.get("plan_progress_ratio"),
            "expected_min_chunks": out.get("expected_min_chunks"),
            "coverage_score": out.get("coverage_score"),
            "confidence_score": out.get("confidence_score"),
            "missing_topics": out.get("missing_topics") or [],
        }

    if normalized_level == 2:
        return {}

    return {}


def limit_evidences(
    evidences: List[Dict[str, Any]],
    max_items: int | None,
) -> List[Dict[str, Any]]:
    limited: List[Dict[str, Any]] = []
    subset = list(evidences) if max_items is None else evidences[: max(max_items, 0)]
    for entry in subset:
        if not isinstance(entry, dict):
            continue
        chunk_id = entry.get("chunk_id")
        content = str(entry.get("content") or "")
        limited.append(
            {
                "chunk_id": chunk_id,
                "source": entry.get("source"),
                "content": content,
                "score": entry.get("score"),
                "provenance": entry.get("provenance"),
            }
        )
    return limited


def limit_highlights(
    highlights: list[Any],
    *,
    max_items: int | None,
) -> list[dict[str, Any]]:
    if not isinstance(highlights, list) or not highlights:
        return []
    subset = highlights if max_items is None else highlights[: max(int(max_items), 0)]
    limited: list[dict[str, Any]] = []
    for item in subset:
        if not isinstance(item, dict):
            continue
        payload: dict[str, Any] = dict(item)
        limited.append(payload)
    return limited
