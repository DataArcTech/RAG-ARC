"""Progress helpers for DeepSearch runs."""
from typing import Any, Dict, Iterable, Optional, Tuple

from config.core.deepsearch import progress_defaults


def compute_deepsearch_progress(
    stage: str,
    *,
    stage_history: Iterable[Dict[str, Any]] | None = None,
    stage_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a stable progress payload for UI consumers.

    Rules:
    - `done` always maps to 100% (API/UI fallback).
    - `reasoned` interpolates percent using step_count + completed_steps when available.
    - Other stages use stable base percentages from config.
    """

    normalized = (stage or "").strip().lower() or "unknown"
    order = list(progress_defaults.STAGE_ORDER)
    step_total = len(order)
    step_index = order.index(normalized) if normalized in order else 0

    plan_total, plan_done = _extract_plan_counts(stage_history=stage_history, stage_record=stage_record)
    percent = _stage_percent(
        normalized,
        step_index=step_index,
        step_total=step_total,
        plan_total=plan_total,
        plan_done=plan_done,
    )

    payload: Dict[str, Any] = {
        "stage": normalized,
        "step_index": step_index,
        "step_total": step_total,
        "percent": int(percent),
    }
    if plan_total is not None:
        payload["plan_step_total"] = int(plan_total)
    if plan_done is not None:
        payload["plan_step_done"] = int(plan_done)
    return payload


def _stage_percent(
    stage: str,
    *,
    step_index: int,
    step_total: int,
    plan_total: Optional[int],
    plan_done: Optional[int],
) -> int:
    base = progress_defaults.STAGE_PERCENT_BASE
    if stage == "done":
        return 100
    if stage == "failed":
        return 100
    if stage == "reasoned":
        start = int(progress_defaults.REASONED_START_PERCENT)
        end = int(progress_defaults.REASONED_END_PERCENT)
        if plan_total and plan_total > 0 and plan_done is not None:
            ratio = max(0.0, min(1.0, float(plan_done) / float(plan_total)))
            return int(round(start + ratio * (end - start)))
        # Fallback: keep reasoned within the expected band (no guessing beyond stage order).
        return max(start, min(end, int(base.get("reasoned", start))))

    if stage in base:
        return int(base[stage])
    # Unknown stages: conservative 0 and monotonic with index when possible.
    if step_total <= 1:
        return 0
    return int(round((max(0, step_index) / float(step_total - 1)) * 100.0))


def _extract_plan_counts(
    *,
    stage_history: Iterable[Dict[str, Any]] | None,
    stage_record: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int]]:
    history = list(stage_history or [])
    plan_total: Optional[int] = None
    plan_done: Optional[int] = None

    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        if numeric < 0:
            return None
        return numeric

    # Prefer planned.stage metadata for step_count when present.
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("stage") or "").strip().lower() != "planned":
            continue
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        plan_total = _coerce_int(meta.get("step_count") or meta.get("total_steps") or meta.get("plan_steps"))
        break

    # completed_steps comes from reasoned stage metadata.
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("stage") or "").strip().lower() != "reasoned":
            continue
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        plan_done = _coerce_int(meta.get("completed_steps"))
        break

    # Stage record may contain fresher values (especially for in-process listeners).
    record_meta = stage_record.get("metadata") if isinstance(stage_record, dict) else None
    if isinstance(record_meta, dict):
        record_total = _coerce_int(record_meta.get("step_count") or record_meta.get("total_steps"))
        record_done = _coerce_int(record_meta.get("completed_steps"))
        if plan_total is None:
            plan_total = record_total
        if plan_done is None:
            plan_done = record_done

    if plan_total is not None and plan_done is not None:
        plan_done = min(plan_done, plan_total)
    return plan_total, plan_done
