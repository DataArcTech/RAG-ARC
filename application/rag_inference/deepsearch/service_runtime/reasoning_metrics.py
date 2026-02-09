"""DeepSearch reasoning metrics (observable + comparable).

These metrics are derived from the reasoning trace and are meant to support
performance comparisons under network jitter:
- tool call counts
- tool latency breakdowns
- cache hit rates (tool-level)
- run-scoped tool memoization stats

They are safe to compute without side effects and contain no evidence text.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _p95(values: List[int]) -> Optional[int]:
    if not values:
        return None
    values = sorted(int(v) for v in values)
    # Nearest-rank percentile.
    idx = int((len(values) - 1) * 0.95)
    idx = min(max(idx, 0), len(values) - 1)
    return int(values[idx])


def _extract_tool_logs(reasoning_steps: Any) -> List[Mapping[str, Any]]:
    logs: List[Mapping[str, Any]] = []
    for step in _as_list(reasoning_steps):
        if not isinstance(step, dict):
            continue
        for entry in _as_list(step.get("tool_logs")):
            if isinstance(entry, dict):
                logs.append(entry)
    return logs


def _extract_tool_results(tool_results: Any) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for entry in _as_list(tool_results):
        if isinstance(entry, dict):
            out.append(entry)
    return out


def summarize_reasoning_trace(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize metrics from a DeepSearch reasoning trace dict."""

    reasoning_steps = trace.get("reasoning_steps")
    tool_logs = _extract_tool_logs(reasoning_steps)

    tool_calls_by_name: Dict[str, int] = {}
    latency_by_name: Dict[str, List[int]] = {}
    latency_total_ms = 0
    for log in tool_logs:
        name = str(log.get("tool_name") or "").strip() or "unknown"
        tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
        latency = log.get("latency_ms")
        try:
            latency_i = int(latency) if latency is not None else None
        except Exception:
            latency_i = None
        if latency_i is not None and latency_i >= 0:
            latency_total_ms += latency_i
            latency_by_name.setdefault(name, []).append(latency_i)

    latency_stats_by_name: Dict[str, Dict[str, Any]] = {}
    for name, values in latency_by_name.items():
        if not values:
            continue
        latency_stats_by_name[name] = {
            "count": len(values),
            "sum_ms": int(sum(values)),
            "mean_ms": float(sum(values)) / float(len(values)),
            "p95_ms": _p95(values),
            "max_ms": int(max(values)),
        }

    tool_results = _extract_tool_results(trace.get("tool_results"))
    cache_hits_by_tool: Dict[str, int] = {}
    cache_misses_by_tool: Dict[str, int] = {}
    cache_hits_total = 0
    cache_misses_total = 0
    explore_actions_total = 0
    explore_actions_by_tool: Dict[str, int] = {}
    explore_cache_hits_total = 0
    explore_cache_misses_total = 0
    explore_cache_hits_by_tool: Dict[str, int] = {}
    explore_cache_misses_by_tool: Dict[str, int] = {}
    for entry in tool_results:
        tool_name = str(entry.get("tool_name") or "").strip() or str(entry.get("tool") or "").strip() or "unknown"
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        diagnostics = result.get("diagnostics") if isinstance(result, dict) else {}
        if not isinstance(diagnostics, dict):
            continue

        # Explore aggregates multiple nested tool calls; count them separately so we can
        # observe toc/tree cache hits even when the model only calls `explore`.
        if tool_name == "explore":
            actions = diagnostics.get("actions")
            if isinstance(actions, list):
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    act_tool = str(action.get("tool") or "").strip() or "unknown"
                    explore_actions_total += 1
                    explore_actions_by_tool[act_tool] = explore_actions_by_tool.get(act_tool, 0) + 1
                    act_diag = action.get("diagnostics")
                    if not isinstance(act_diag, dict):
                        continue
                    act_cache = act_diag.get("cache")
                    if not isinstance(act_cache, dict):
                        continue
                    act_hit = act_cache.get("hit")
                    if act_hit is True:
                        explore_cache_hits_total += 1
                        explore_cache_hits_by_tool[act_tool] = explore_cache_hits_by_tool.get(act_tool, 0) + 1
                    elif act_hit is False:
                        explore_cache_misses_total += 1
                        explore_cache_misses_by_tool[act_tool] = explore_cache_misses_by_tool.get(act_tool, 0) + 1

        cache = diagnostics.get("cache")
        if isinstance(cache, dict):
            hit = cache.get("hit")
            if hit is True:
                cache_hits_total += 1
                cache_hits_by_tool[tool_name] = cache_hits_by_tool.get(tool_name, 0) + 1
            elif hit is False:
                cache_misses_total += 1
                cache_misses_by_tool[tool_name] = cache_misses_by_tool.get(tool_name, 0) + 1

    memo_stats = trace.get("tool_memoization") if isinstance(trace.get("tool_memoization"), dict) else {}

    evidences = trace.get("evidences")
    page_evidence_items = 0
    for ev in _as_list(evidences):
        if not isinstance(ev, dict):
            continue
        if str(ev.get("source") or "").strip().lower() == "read.pages":
            page_evidence_items += 1

    return {
        "tool_calls_total": int(sum(tool_calls_by_name.values())),
        "tool_calls_by_name": dict(sorted(tool_calls_by_name.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tool_latency_total_ms": int(latency_total_ms),
        "tool_latency_by_name": latency_stats_by_name,
        "cache_hits_total": int(cache_hits_total),
        "cache_misses_total": int(cache_misses_total),
        "cache_hits_by_tool": dict(sorted(cache_hits_by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
        "cache_misses_by_tool": dict(sorted(cache_misses_by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
        "explore_actions_total": int(explore_actions_total),
        "explore_actions_by_tool": dict(sorted(explore_actions_by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
        "explore_cache_hits_total": int(explore_cache_hits_total),
        "explore_cache_misses_total": int(explore_cache_misses_total),
        "explore_cache_hits_by_tool": dict(sorted(explore_cache_hits_by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
        "explore_cache_misses_by_tool": dict(sorted(explore_cache_misses_by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tool_memoization": memo_stats,
        "primary_page_evidence_items": int(page_evidence_items),
    }


__all__ = ["summarize_reasoning_trace"]
