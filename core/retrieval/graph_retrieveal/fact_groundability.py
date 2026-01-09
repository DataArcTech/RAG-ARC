from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FactGroundabilityConfig:
    enabled: bool
    mode: str
    dense_top_k: int
    min_overlap_count: int
    min_overlap_ratio: float
    soft_min_weight: float
    soft_gamma: float
    keep_missing_provenance: bool
    missing_provenance_weight: float


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def extract_fact_source_chunk_ids(*, docstore: Mapping[str, Any], fact_id: str) -> list[str]:
    chunk = docstore.get(fact_id)
    meta = getattr(chunk, "metadata", None)
    if not isinstance(meta, dict):
        return []
    return _coerce_str_list(meta.get("source_chunk_ids"))


def compute_overlap(*, source_chunk_ids: Sequence[str], retrieved_chunk_ids: set[str]) -> tuple[int, float | None]:
    source = {str(x) for x in source_chunk_ids if str(x)}
    if not source:
        return 0, None
    overlap = source.intersection(retrieved_chunk_ids)
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / max(1, len(source))
    return overlap_count, overlap_ratio


def apply_groundability(
    *,
    cfg: FactGroundabilityConfig,
    scores: np.ndarray,
    fact_ids: Sequence[str],
    docstore: Mapping[str, Any],
    dense_top_chunk_ids: set[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not cfg.enabled or scores is None or len(scores) == 0 or not fact_ids:
        return scores, list(fact_ids), {"enabled": bool(cfg.enabled), "applied": False}

    if len(scores) != len(fact_ids):
        raise ValueError(f"fact score/id length mismatch: {len(scores)} != {len(fact_ids)}")

    applied = 0
    kept_scores: list[float] = []
    kept_ids: list[str] = []
    dropped = 0
    missing = 0

    mode = str(cfg.mode or "").strip().lower()
    if mode not in {"hard_filter", "soft_penalty"}:
        raise ValueError(f"Unknown fact_groundability mode: {cfg.mode!r}")

    for score, fact_id in zip(scores.tolist(), fact_ids):
        source_chunk_ids = extract_fact_source_chunk_ids(docstore=docstore, fact_id=str(fact_id))
        overlap_count, overlap_ratio = compute_overlap(
            source_chunk_ids=source_chunk_ids,
            retrieved_chunk_ids=dense_top_chunk_ids,
        )

        if overlap_ratio is None:
            missing += 1
            if not cfg.keep_missing_provenance and mode == "hard_filter":
                dropped += 1
                continue
            factor = float(cfg.missing_provenance_weight)
            kept_scores.append(float(score) * factor)
            kept_ids.append(str(fact_id))
            applied += 1
            continue

        if mode == "hard_filter":
            if overlap_count < int(cfg.min_overlap_count) or overlap_ratio < float(cfg.min_overlap_ratio):
                dropped += 1
                continue
            kept_scores.append(float(score))
            kept_ids.append(str(fact_id))
            applied += 1
            continue

        soft_min = float(cfg.soft_min_weight)
        gamma = float(cfg.soft_gamma)
        ratio = float(overlap_ratio)
        factor = soft_min + (1.0 - soft_min) * (ratio ** gamma)
        kept_scores.append(float(score) * factor)
        kept_ids.append(str(fact_id))
        applied += 1

    if not kept_ids:
        return scores, list(fact_ids), {"enabled": True, "applied": False, "reason": "all_dropped", "dropped": dropped}

    out_scores = np.array(kept_scores, dtype=scores.dtype)
    return out_scores, kept_ids, {
        "enabled": True,
        "applied": True,
        "mode": mode,
        "kept": len(kept_ids),
        "dropped": dropped,
        "missing_provenance": missing,
    }


__all__ = [
    "FactGroundabilityConfig",
    "apply_groundability",
    "compute_overlap",
    "extract_fact_source_chunk_ids",
]

