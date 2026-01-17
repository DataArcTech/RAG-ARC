from collections import Counter
from typing import Any, Optional, Sequence, Tuple


def pick_dense_top_file_id(
    source_file_ids_ranked: Sequence[Optional[str]],
    *,
    top_k: int,
) -> Tuple[Optional[str], float, int]:
    """
    Pick the most frequent `source_file_id` from the top-K ranked dense chunks.

    Returns:
      (top_file_id, top_ratio, top_count)
    """
    top_k = max(0, int(top_k))
    ids = [str(x) for x in source_file_ids_ranked[:top_k] if x]
    if not ids:
        return None, 0.0, 0
    counts = Counter(ids)
    top_file_id, top_count = counts.most_common(1)[0]
    ratio = float(top_count) / float(len(ids))
    return top_file_id, ratio, int(top_count)

def dense_file_distribution_stats(
    source_file_ids_ranked: Sequence[Optional[str]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """
    Compute simple stats over `source_file_id` distribution in dense top-K hits.

    This is used to gate an optional file-level prior. We keep the output JSON-like
    so callers can log/attach it for observability.
    """
    top_k = max(0, int(top_k))
    ids = [str(x) for x in source_file_ids_ranked[:top_k] if x]
    if not ids:
        return {
            "total": 0,
            "unique": 0,
            "top_file_id": None,
            "top_count": 0,
            "top_ratio": 0.0,
            "second_file_id": None,
            "second_count": 0,
            "second_ratio": 0.0,
        }

    counts = Counter(ids)
    most = counts.most_common(2)
    top_file_id, top_count = most[0]
    second_file_id = None
    second_count = 0
    if len(most) > 1:
        second_file_id, second_count = most[1]

    total = len(ids)
    return {
        "total": int(total),
        "unique": int(len(counts)),
        "top_file_id": top_file_id,
        "top_count": int(top_count),
        "top_ratio": float(top_count) / float(total) if total else 0.0,
        "second_file_id": second_file_id,
        "second_count": int(second_count),
        "second_ratio": float(second_count) / float(total) if total else 0.0,
    }

def should_apply_dense_file_prior(
    *,
    top_ratio: float,
    second_ratio: float,
    min_ratio: float,
    min_margin: float,
) -> bool:
    """
    Decide whether the dense-file prior should activate based on top-K distribution.

    We require:
    - top_ratio >= min_ratio
    - (top_ratio - second_ratio) >= min_margin
    """
    try:
        top = float(top_ratio)
        second = float(second_ratio)
        ratio_th = float(min_ratio)
        margin_th = float(min_margin)
    except Exception:  # noqa: BLE001
        return False
    return (top >= ratio_th) and ((top - second) >= margin_th)

def compute_dense_file_prior_multipliers(
    source_file_ids_ranked: Sequence[Optional[str]],
    *,
    top_k: int,
    max_files: int,
    min_ratio: float,
    min_margin: float,
    multiplier: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Compute per-file multipliers for an optional dense-derived file prior.

    When activated, we boost PPR reset weights for chunks that belong to the most frequent
    file(s) in dense top-K hits. This keeps retrieval stable for file-scoped queries while
    still allowing multi-file coverage for mixed-intent queries.
    """
    top_k = max(0, int(top_k))
    max_files = max(1, int(max_files or 1))
    base_multiplier = float(multiplier or 1.0)

    ids = [str(x) for x in source_file_ids_ranked[:top_k] if x]
    stats = dense_file_distribution_stats(ids, top_k=len(ids))
    if not ids:
        stats["applied"] = False
        stats["prior_files"] = []
        return {}, stats

    counts = Counter(ids)
    most = counts.most_common(max_files)
    top_ratio = float(stats.get("top_ratio") or 0.0)
    second_ratio = float(stats.get("second_ratio") or 0.0)

    applied = should_apply_dense_file_prior(
        top_ratio=top_ratio,
        second_ratio=second_ratio,
        min_ratio=min_ratio,
        min_margin=min_margin,
    )
    stats["applied"] = bool(applied)

    if not applied or top_ratio <= 0.0 or base_multiplier <= 0.0:
        stats["prior_files"] = []
        return {}, stats

    # Scale additional files by their relative share so we don't over-boost weak secondary files.
    out: dict[str, float] = {}
    prior_files: list[dict[str, Any]] = []
    for file_id, count in most:
        ratio_i = float(count) / float(len(ids)) if ids else 0.0
        # Ensure top file gets the configured multiplier; others scale down proportionally.
        eff = 1.0 + (base_multiplier - 1.0) * (ratio_i / top_ratio)
        eff = float(max(1.0, eff))
        out[str(file_id)] = eff
        prior_files.append({"file_id": str(file_id), "ratio": ratio_i, "multiplier": eff})
    stats["prior_files"] = prior_files
    return out, stats


__all__ = [
    "compute_dense_file_prior_multipliers",
    "dense_file_distribution_stats",
    "pick_dense_top_file_id",
    "should_apply_dense_file_prior",
]
