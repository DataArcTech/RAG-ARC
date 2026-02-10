from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(aa) + 1e-10)
    nb = float(np.linalg.norm(bb) + 1e-10)
    return float(np.dot(aa, bb) / (na * nb))


@dataclass
class GreedyCluster:
    indices: list[int]
    centroid: np.ndarray  # normalized
    count: int


def greedy_cluster_by_centroid(
    embeddings: np.ndarray,
    *,
    min_sim: float,
) -> list[GreedyCluster]:
    """Greedy, deterministic clustering by centroid similarity.

    Algorithm
    - Iterate embeddings in input order.
    - Assign point to the first cluster whose centroid cosine sim >= min_sim.
    - Otherwise, create a new cluster with this point.

    Notes
    - Complexity is O(N*K) where K is number of clusters (usually small).
    - This is conservative: it tends to create new clusters only when a point is clearly dissimilar.
    """
    arr = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    if arr.size == 0:
        return []

    clusters: list[GreedyCluster] = []
    for i in range(arr.shape[0]):
        v = arr[i]
        assigned = False
        for c in clusters:
            sim = float(np.dot(v, c.centroid))
            if sim >= float(min_sim):
                c.indices.append(i)
                # online mean then renormalize
                c.count += 1
                c.centroid = normalize_rows((c.centroid * float(c.count - 1) + v) / float(c.count))[0]
                assigned = True
                break
        if not assigned:
            clusters.append(GreedyCluster(indices=[i], centroid=v.copy(), count=1))
    return clusters

