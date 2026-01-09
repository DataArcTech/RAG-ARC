from typing import Any, Dict, List, Tuple


def filter_similarity_neighbors(
    neighbors: List[Dict[str, Any]],
    *,
    hop: int,
    relation_name: str,
    max_hops: int,
    min_similarity: float,
    max_per_node: int,
) -> List[Dict[str, Any]]:
    """Filter SIMILAR_TO-style neighbors to reduce multi-hop error amplification."""

    if not neighbors:
        return []

    rel = str(relation_name or "SIMILAR_TO").strip().upper() or "SIMILAR_TO"
    max_hops = max(0, int(max_hops))
    if hop >= max_hops:
        return [n for n in neighbors if str((n or {}).get("relation_type") or "").strip().upper() != rel]

    threshold = float(min_similarity)
    keep: List[Dict[str, Any]] = []
    similar: List[Dict[str, Any]] = []
    for entry in neighbors:
        if not isinstance(entry, dict):
            continue
        rtype = str(entry.get("relation_type") or "").strip().upper()
        if rtype != rel:
            keep.append(entry)
            continue
        sim_raw = entry.get("similarity")
        try:
            sim = float(sim_raw) if sim_raw is not None else None
        except Exception:
            sim = None
        if sim is not None and sim < threshold:
            continue
        similar.append(entry)

    similar.sort(key=lambda item: float(item.get("similarity") or 0.0), reverse=True)
    cap = max(0, int(max_per_node))
    if cap == 0:
        similar = []
    elif cap:
        similar = similar[:cap]
    return keep + similar


__all__ = ["filter_similarity_neighbors"]


def filter_similarity_neighbor_tuples(
    neighbors: List[Tuple[str, float, str]],
    *,
    hop: int,
    relation_name: str,
    max_hops: int,
    min_similarity: float,
    max_per_node: int,
) -> List[Tuple[str, float, str]]:
    """Tuple variant for (neighbor_id, weight_or_similarity, relation_type)."""

    if not neighbors:
        return []

    rel = str(relation_name or "SIMILAR_TO").strip().upper() or "SIMILAR_TO"
    max_hops = max(0, int(max_hops))
    cap = max(0, int(max_per_node))
    threshold = float(min_similarity)

    keep: List[Tuple[str, float, str]] = []
    similar: List[Tuple[str, float, str]] = []
    for neighbor_id, weight, rtype in neighbors:
        kind = str(rtype or "").strip().upper()
        if kind != rel:
            keep.append((neighbor_id, float(weight), rtype))
            continue
        if hop >= max_hops:
            continue
        if float(weight) < threshold:
            continue
        similar.append((neighbor_id, float(weight), rtype))

    similar.sort(key=lambda item: item[1], reverse=True)
    if cap == 0:
        similar = []
    elif cap:
        similar = similar[:cap]
    return keep + similar


__all__ = ["filter_similarity_neighbors", "filter_similarity_neighbor_tuples"]
