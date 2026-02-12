"""Web search aggregation utilities (source-level de-biasing)."""
import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence
from urllib.parse import urlparse

from .tavily_client import TavilySearchResult


@dataclass(frozen=True)
class WebSearchAggregationStats:
    group_by: str
    total_in: int
    total_out: int
    max_groups: int
    max_items_per_group: int
    groups: List[dict]


def _normalize_group_by(group_by: str | None) -> str:
    token = str(group_by or "domain").strip().lower()
    if token in {"domain", "url", "provider"}:
        return token
    return "domain"


def _domain_from_url(url: str | None) -> str:
    raw = str(url or "").strip()
    if not raw:
        return "unknown"
    parsed = urlparse(raw)
    host = parsed.netloc or parsed.path
    host = host.split("/")[0].strip()
    if not host:
        return "unknown"
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    host = host.split(":", 1)[0].strip()
    return host or "unknown"


def _score(value: float | None) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


def aggregate_tavily_results(
    results: Sequence[TavilySearchResult],
    *,
    group_by: str = "domain",
    max_groups: int = 3,
    max_items_per_group: int = 2,
) -> tuple[List[TavilySearchResult], WebSearchAggregationStats]:
    """Aggregate Tavily results by source to reduce single-source dominance.

    Scoring: sum(score) / sqrt(n + 1)
    """
    items = list(results or [])
    total_in = len(items)
    if total_in == 0:
        stats = WebSearchAggregationStats(
            group_by=_normalize_group_by(group_by),
            total_in=0,
            total_out=0,
            max_groups=int(max_groups),
            max_items_per_group=int(max_items_per_group),
            groups=[],
        )
        return [], stats

    mg = int(max_groups or 0)
    mp = int(max_items_per_group or 0)
    if mg <= 0 or mp <= 0:
        stats = WebSearchAggregationStats(
            group_by=_normalize_group_by(group_by),
            total_in=total_in,
            total_out=total_in,
            max_groups=mg,
            max_items_per_group=mp,
            groups=[],
        )
        return items, stats

    grouping = _normalize_group_by(group_by)

    def _group_key(item: TavilySearchResult) -> str:
        if grouping == "url":
            return str(item.url or "unknown").strip().lower() or "unknown"
        if grouping == "provider":
            return "tavily"
        return _domain_from_url(item.url)

    grouped: dict[str, list[tuple[int, float, TavilySearchResult]]] = {}
    for idx, item in enumerate(items):
        key = _group_key(item)
        grouped.setdefault(key, []).append((idx, _score(item.score), item))

    group_summaries: list[tuple[float, str, int]] = []
    for key, members in grouped.items():
        n = len(members)
        s = sum(max(0.0, score) for _idx, score, _item in members)
        group_score = s / math.sqrt(n + 1.0)
        group_summaries.append((float(group_score), key, n))

    group_summaries.sort(key=lambda t: (-t[0], t[1]))
    kept_groups = group_summaries[: min(mg, len(group_summaries))]
    kept_keys = {key for _score, key, _n in kept_groups}

    out: List[TavilySearchResult] = []
    groups_info: List[dict] = []
    for score, key, n in kept_groups:
        members = grouped.get(key) or []
        members_sorted = sorted(members, key=lambda t: (-t[1], t[0]))
        selected = members_sorted[: min(mp, len(members_sorted))]
        out.extend([item for _idx, _score, item in selected])
        groups_info.append(
            {
                "key": key,
                "group_score": score,
                "items_in": n,
                "items_out": len(selected),
            }
        )

    stats = WebSearchAggregationStats(
        group_by=grouping,
        total_in=total_in,
        total_out=len(out),
        max_groups=mg,
        max_items_per_group=mp,
        groups=groups_info,
    )
    return out, stats
