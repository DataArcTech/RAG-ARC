from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence, Tuple


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


__all__ = ["pick_dense_top_file_id"]

