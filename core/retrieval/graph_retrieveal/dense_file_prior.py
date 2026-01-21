from collections import Counter
from typing import Any, Mapping, Optional, Sequence, Tuple

from core.utils.index_text_augmentation import extract_title_from_filename


def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    # Common CJK ranges (Han + extensions) to keep tokenization simple and dependency-free.
    return (
        0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
    )


def _lexical_tokens(text: str) -> set[str]:
    """
    Lightweight, domain-agnostic tokenization for gating heuristics.

    - CJK: use character bigrams for better specificity than unigrams.
    - ASCII words/digits: split into alnum runs.
    """
    raw = str(text or "").strip()
    if not raw:
        return set()

    tokens: set[str] = set()
    cjk_buf: list[str] = []
    ascii_buf: list[str] = []

    def flush_ascii() -> None:
        nonlocal ascii_buf
        if not ascii_buf:
            return
        tok = "".join(ascii_buf).strip().lower()
        ascii_buf = []
        if not tok:
            return
        # Avoid ultra-short noise tokens ("a", "i") unless digits are present.
        if len(tok) >= 2 or any(ch.isdigit() for ch in tok):
            tokens.add(tok)

    def flush_cjk() -> None:
        nonlocal cjk_buf
        if not cjk_buf:
            return
        if len(cjk_buf) == 1:
            tokens.add(cjk_buf[0])
            cjk_buf = []
            return
        for i in range(len(cjk_buf) - 1):
            tokens.add(cjk_buf[i] + cjk_buf[i + 1])
        cjk_buf = []

    for ch in raw:
        if _is_cjk(ch):
            flush_ascii()
            cjk_buf.append(ch)
            continue
        # ASCII word characters.
        if ch.isascii() and ch.isalnum():
            flush_cjk()
            ascii_buf.append(ch)
            continue
        # Delimiters/punctuations.
        flush_ascii()
        flush_cjk()

    flush_ascii()
    flush_cjk()
    return tokens


def file_title_lexical_coverage(
    query_variants: Sequence[str],
    *,
    filename: str,
) -> tuple[float, int, str]:
    """
    Compute how much of the file title is "covered" by query tokens.

    Returns:
      (best_coverage, best_overlap_tokens, title)
    where
      coverage = |tokens(title) ∩ tokens(query)| / max(1, |tokens(title)|)
    """
    title = extract_title_from_filename(str(filename or ""))
    title_tokens = _lexical_tokens(title)
    if not title_tokens:
        return 0.0, 0, title

    best_cov = 0.0
    best_overlap = 0
    for q in query_variants:
        q_tokens = _lexical_tokens(str(q or ""))
        if not q_tokens:
            continue
        overlap = len(title_tokens.intersection(q_tokens))
        cov = float(overlap) / float(len(title_tokens))
        if cov > best_cov:
            best_cov = cov
            best_overlap = overlap
    return float(best_cov), int(best_overlap), title


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
    max_second_ratio: float | None = None,
) -> bool:
    """
    Decide whether the dense-file prior should activate based on top-K distribution.

    We require:
    - top_ratio >= min_ratio
    - (top_ratio - second_ratio) >= min_margin
    - if max_second_ratio is set: second_ratio <= max_second_ratio
    """
    try:
        top = float(top_ratio)
        second = float(second_ratio)
        ratio_th = float(min_ratio)
        margin_th = float(min_margin)
        max_second = float(max_second_ratio) if max_second_ratio is not None else None
    except Exception:  # noqa: BLE001
        return False
    if not ((top >= ratio_th) and ((top - second) >= margin_th)):
        return False
    if max_second is not None and second > max_second:
        return False
    return True

def compute_dense_file_prior_multipliers(
    source_file_ids_ranked: Sequence[Optional[str]],
    *,
    top_k: int,
    max_files: int,
    min_ratio: float,
    min_margin: float,
    max_second_ratio: float | None = None,
    multiplier: float,
    # Optional: extra gate using query<->filename lexical consistency to avoid "confidently wrong" priors.
    query_variants: Sequence[str] | None = None,
    filenames_by_file_id: Mapping[str, str] | None = None,
    lexical_min_top_ratio: float | None = None,
    lexical_min_title_coverage: float | None = None,
    lexical_min_overlap_tokens: int | None = None,
    lexical_min_margin: float | None = None,
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
        max_second_ratio=max_second_ratio,
    )
    stats["applied"] = bool(applied)

    if not applied or top_ratio <= 0.0 or base_multiplier <= 0.0:
        stats["prior_files"] = []
        return {}, stats

    # Optional lexical gate: require that the dominant file's title is actually mentioned by the query.
    # This is domain-agnostic and helps avoid PPR drift when embeddings are "confidently wrong".
    lexical_gate_enabled = (
        applied
        and query_variants is not None
        and filenames_by_file_id is not None
        and lexical_min_title_coverage is not None
        and lexical_min_overlap_tokens is not None
    )
    if lexical_gate_enabled:
        try:
            min_top_ratio = float(lexical_min_top_ratio) if lexical_min_top_ratio is not None else None
            cov_th = float(lexical_min_title_coverage)
            overlap_th = int(lexical_min_overlap_tokens)
            margin_th = float(lexical_min_margin) if lexical_min_margin is not None else 0.0
        except Exception:  # noqa: BLE001
            min_top_ratio = None
            cov_th = 0.0
            overlap_th = 0
            margin_th = 0.0

        if min_top_ratio is not None and top_ratio < min_top_ratio:
            stats["lexical_gate"] = {
                "enabled": True,
                "skipped": True,
                "reason": "top_ratio_below_min_top_ratio",
                "min_top_ratio": float(min_top_ratio),
                "top_ratio": float(top_ratio),
            }
            # Do not block the prior when dense evidence is not strongly concentrated; this is often a multi-file query.
        else:
            # Compute lexical coverage for the top-2 file candidates.
            top_file_id = str(stats.get("top_file_id") or "")
            second_file_id = str(stats.get("second_file_id") or "")
            top_fn = str(filenames_by_file_id.get(top_file_id) or "")
            second_fn = str(filenames_by_file_id.get(second_file_id) or "")

            top_cov, top_overlap, top_title = file_title_lexical_coverage(query_variants, filename=top_fn)
            second_cov, _second_overlap, second_title = (
                file_title_lexical_coverage(query_variants, filename=second_fn) if second_fn else (0.0, 0, "")
            )

            stats["lexical_gate"] = {
                "enabled": True,
                "min_top_ratio": float(min_top_ratio) if min_top_ratio is not None else None,
                "min_title_coverage": float(cov_th),
                "min_overlap_tokens": int(overlap_th),
                "min_margin": float(margin_th),
                "top": {
                    "file_id": top_file_id or None,
                    "title": top_title,
                    "coverage": float(top_cov),
                    "overlap_tokens": int(top_overlap),
                },
                "second": {
                    "file_id": second_file_id or None,
                    "title": second_title,
                    "coverage": float(second_cov),
                },
            }

            passed = (top_cov >= cov_th) and (top_overlap >= overlap_th) and ((top_cov - second_cov) >= margin_th)
            if not passed:
                stats["applied"] = False
                stats["prior_files"] = []
                stats["lexical_gate"]["passed"] = False
                return {}, stats

            stats["lexical_gate"]["passed"] = True

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
