from dataclasses import dataclass


@dataclass(frozen=True)
class RRFInput:
    id: str
    rank: int


def rrf_score(rank: int, *, rrf_k: int, weight: float) -> float:
    if rank <= 0:
        return 0.0
    return float(weight) / float(rrf_k + rank)


def fuse_rrf(
    *,
    sources: dict[str, list[RRFInput]],
    rrf_k: int,
    weights: dict[str, float],
    both_sources_bonus: float = 0.0,
    k: int,
) -> list[str]:
    rank_maps: dict[str, dict[str, int]] = {name: {item.id: item.rank for item in items} for name, items in sources.items()}
    all_ids: set[str] = set()
    for ranks in rank_maps.values():
        all_ids |= set(ranks.keys())

    fused: dict[str, float] = {}
    for cid in all_ids:
        present = 0
        score = 0.0
        for source_name, ranks in rank_maps.items():
            if cid not in ranks:
                continue
            present += 1
            score += rrf_score(ranks[cid], rrf_k=rrf_k, weight=float(weights.get(source_name, 1.0)))
        if present >= 2 and both_sources_bonus > 0:
            score += float(both_sources_bonus)
        fused[cid] = score

    ranked_ids = sorted(all_ids, key=lambda cid: (fused.get(cid, 0.0), cid), reverse=True)[: int(k)]
    return ranked_ids


def fuse_rrf_with_scores(
    *,
    sources: dict[str, list[RRFInput]],
    rrf_k: int,
    weights: dict[str, float],
    both_sources_bonus: float = 0.0,
    k: int,
) -> tuple[list[str], dict[str, float]]:
    rank_maps: dict[str, dict[str, int]] = {name: {item.id: item.rank for item in items} for name, items in sources.items()}
    all_ids: set[str] = set()
    for ranks in rank_maps.values():
        all_ids |= set(ranks.keys())

    fused: dict[str, float] = {}
    for cid in all_ids:
        present = 0
        score = 0.0
        for source_name, ranks in rank_maps.items():
            if cid not in ranks:
                continue
            present += 1
            score += rrf_score(ranks[cid], rrf_k=rrf_k, weight=float(weights.get(source_name, 1.0)))
        if present >= 2 and both_sources_bonus > 0:
            score += float(both_sources_bonus)
        fused[cid] = score

    ranked_ids = sorted(all_ids, key=lambda cid: (fused.get(cid, 0.0), cid), reverse=True)[: int(k)]
    return ranked_ids, fused
