"""Gradient truncation for candidate lists.

Given a score-descending list:
- Always keep the first `mink` items (if available).
- Then keep adding items while the next score is not a sharp drop:
    score_i >= score_{i-1} * g

This avoids hard-coding an absolute score cutoff and adapts to the local score distribution.
"""
from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class GradientSelect:
    mink: int
    g: float

    def normalized(self) -> "GradientSelect":
        mink = max(0, int(self.mink))
        g = float(self.g)
        # Clamp to [0, 1]. Values > 1 can cause "keep everything" behavior.
        g = min(1.0, max(0.0, g))
        return GradientSelect(mink=mink, g=g)


def truncate_indices(scores_desc: Sequence[float], *, policy: GradientSelect) -> list[int]:
    policy = policy.normalized()
    if not scores_desc:
        return []

    out: list[int] = []
    mink = min(len(scores_desc), policy.mink)
    if mink > 0:
        out.extend(range(mink))
        prev = float(scores_desc[mink - 1])
        start = mink
    else:
        out.append(0)
        prev = float(scores_desc[0])
        start = 1

    # Defensive: negative scores break the multiplicative drop test.
    if prev < 0:
        return out

    for i in range(start, len(scores_desc)):
        cur = float(scores_desc[i])
        if cur < 0:
            break
        if cur >= prev * policy.g:
            out.append(i)
            prev = cur
        else:
            break
    return out


def truncate(items_desc: Sequence[T], scores_desc: Sequence[float], *, policy: GradientSelect) -> list[T]:
    if len(items_desc) != len(scores_desc):
        raise ValueError("items_desc and scores_desc must have the same length")
    idxs = truncate_indices(scores_desc, policy=policy)
    return [items_desc[i] for i in idxs]

