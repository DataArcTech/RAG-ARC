from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class QuotaAllocation:
    quotas: List[int]
    normalized_weights: List[float]


def allocate_quotas(total_k: int, weights: Iterable[float]) -> QuotaAllocation:
    """
    Allocate integer quotas from non-negative weights.

    Rules:
    - weight <= 0 -> quota 0
    - if total_k >= number_of_active_backends: give each active backend at least 1 (coverage floor)
    - allocate remaining budget proportionally, then distribute remainders by largest fractional part
    """
    total_k = max(0, int(total_k))
    ws = [float(w) for w in weights]
    active = [i for i, w in enumerate(ws) if w > 0]

    if total_k <= 0 or not ws:
        return QuotaAllocation(quotas=[0 for _ in ws], normalized_weights=[0.0 for _ in ws])

    # Normalize weights across active backends for reporting/debugging only.
    s = sum(ws[i] for i in active) if active else 0.0
    norm = [0.0 for _ in ws]
    if s > 0:
        for i in active:
            norm[i] = ws[i] / s

    quotas = [0 for _ in ws]
    if not active:
        return QuotaAllocation(quotas=quotas, normalized_weights=norm)

    # Coverage floor: 1 each when budget allows.
    remaining = total_k
    if total_k >= len(active):
        for i in active:
            quotas[i] = 1
        remaining = total_k - len(active)

    if remaining <= 0:
        return QuotaAllocation(quotas=quotas, normalized_weights=norm)

    # Proportional allocation for the remaining budget.
    s_active = sum(ws[i] for i in active)
    if s_active <= 0:
        # If all active weights are effectively zero (shouldn't happen), distribute evenly.
        per = remaining // len(active)
        for i in active:
            quotas[i] += per
        rest = remaining - per * len(active)
        for i in active[:rest]:
            quotas[i] += 1
        return QuotaAllocation(quotas=quotas, normalized_weights=norm)

    exact: List[Tuple[int, float]] = []
    used = 0
    for i in active:
        want = remaining * (ws[i] / s_active)
        base = int(want)
        frac = float(want) - base
        quotas[i] += base
        used += base
        exact.append((i, frac))

    leftover = remaining - used
    if leftover > 0:
        exact.sort(key=lambda x: x[1], reverse=True)
        for i, _ in exact[:leftover]:
            quotas[i] += 1

    return QuotaAllocation(quotas=quotas, normalized_weights=norm)

