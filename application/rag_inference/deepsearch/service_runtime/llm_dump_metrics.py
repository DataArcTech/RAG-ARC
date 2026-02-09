"""Optional metrics derived from DEEPSEARCH_LLM_DUMP_PATH JSONL.

This is an observability helper:
- It must not change DeepSearch runtime behavior.
- It must never raise (callers should best-effort).
- It contains no evidence text in the summarized output.
"""
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _p95(values: List[int]) -> Optional[int]:
    if not values:
        return None
    values = sorted(int(v) for v in values)
    idx = int((len(values) - 1) * 0.95)
    idx = min(max(idx, 0), len(values) - 1)
    return int(values[idx])


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload


def summarize_llm_dump_metrics(*, path: str, run_id: str) -> Dict[str, Any]:
    """Summarize per-run LLM metrics from a JSONL dump file."""

    dump_path = Path(str(path)).expanduser()
    if not dump_path.exists() or not dump_path.is_file():
        return {"enabled": False, "reason": "dump_path_missing", "path": str(dump_path)}
    token = str(run_id or "").strip()
    if not token:
        return {"enabled": False, "reason": "missing_run_id", "path": str(dump_path)}

    requests = 0
    responses = 0
    errors = 0
    elapsed_all: List[int] = []
    calls_by_model: Dict[str, int] = {}
    errors_by_model: Dict[str, int] = {}
    latency_by_model: Dict[str, List[int]] = {}
    calls_by_context: Dict[str, int] = {}
    errors_by_context: Dict[str, int] = {}

    lines_scanned = 0
    lines_matched = 0
    for ev in _iter_jsonl(dump_path):
        lines_scanned += 1
        if str(ev.get("run_id") or "").strip() != token:
            continue
        lines_matched += 1
        kind = str(ev.get("event") or "").strip()
        model = str(ev.get("model") or "").strip() or "unknown"
        ctx = str(ev.get("warn_context") or "").strip() or "unknown"
        if kind == "llm.request":
            requests += 1
            calls_by_model[model] = calls_by_model.get(model, 0) + 1
            calls_by_context[ctx] = calls_by_context.get(ctx, 0) + 1
            continue
        if kind == "llm.response":
            responses += 1
        elif kind == "llm.error":
            errors += 1
            errors_by_model[model] = errors_by_model.get(model, 0) + 1
            errors_by_context[ctx] = errors_by_context.get(ctx, 0) + 1
        else:
            continue

        elapsed = ev.get("elapsed_ms")
        try:
            elapsed_i = int(elapsed) if elapsed is not None else None
        except Exception:
            elapsed_i = None
        if elapsed_i is not None and elapsed_i >= 0:
            elapsed_all.append(elapsed_i)
            latency_by_model.setdefault(model, []).append(elapsed_i)

    latency_stats = {
        "count": len(elapsed_all),
        "sum_ms": int(sum(elapsed_all)),
        "mean_ms": (float(sum(elapsed_all)) / float(len(elapsed_all))) if elapsed_all else None,
        "p95_ms": _p95(elapsed_all),
        "max_ms": int(max(elapsed_all)) if elapsed_all else None,
    }

    latency_stats_by_model: Dict[str, Dict[str, Any]] = {}
    for model, values in latency_by_model.items():
        if not values:
            continue
        latency_stats_by_model[model] = {
            "count": len(values),
            "sum_ms": int(sum(values)),
            "mean_ms": float(sum(values)) / float(len(values)),
            "p95_ms": _p95(values),
            "max_ms": int(max(values)),
        }

    ok_calls = int(responses)
    total_finished = int(responses + errors)
    failure_rate = (float(errors) / float(total_finished)) if total_finished else None
    return {
        "enabled": True,
        "path": str(dump_path),
        "run_id": token,
        "lines_scanned": int(lines_scanned),
        "lines_matched": int(lines_matched),
        "requests": int(requests),
        "responses": int(responses),
        "errors": int(errors),
        "ok_calls": ok_calls,
        "total_finished_calls": total_finished,
        "failure_rate": failure_rate,
        "latency": latency_stats,
        "calls_by_model": dict(sorted(calls_by_model.items(), key=lambda kv: (-kv[1], kv[0]))),
        "errors_by_model": dict(sorted(errors_by_model.items(), key=lambda kv: (-kv[1], kv[0]))),
        "latency_by_model": latency_stats_by_model,
        "calls_by_warn_context": dict(sorted(calls_by_context.items(), key=lambda kv: (-kv[1], kv[0]))),
        "errors_by_warn_context": dict(sorted(errors_by_context.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


__all__ = ["summarize_llm_dump_metrics"]
