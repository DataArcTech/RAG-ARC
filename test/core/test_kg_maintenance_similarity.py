import numpy as np

from core.knowledge_graph.maintenance.similarity import greedy_cluster_by_centroid


def test_greedy_cluster_by_centroid_splits_distant_points() -> None:
    embs = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    clusters = greedy_cluster_by_centroid(embs, min_sim=0.8)
    assert len(clusters) == 2
    assert clusters[0].indices == [0, 1]
    assert clusters[1].indices == [2]

