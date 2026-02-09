import os

import numpy as np
import pytest

from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from core.knowledge_graph.maintenance.similarity import greedy_cluster_by_centroid, normalize_rows


@pytest.mark.skipif(
    os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1",
    reason="integration test opt-in: set RUN_RAGARC_INTEGRATION_TESTS=1",
)
def test_live_embedding_can_separate_two_senses_for_same_surface_name() -> None:
    """
    E2E-ish smoke test: call a real embedding model and check that two very different
    contexts for the same surface string are not forced into a single cluster.

    Notes:
    - This does not require Neo4j; it exercises the deterministic clustering core.
    - If this fails for a particular embedding backend, L1 disambiguation quality will suffer.
    """

    emb = OpenAIEmbeddingConfig().build()

    # Same surface string ("长城") but two clearly different real-world meanings:
    # - A company brand ("长城汽车")
    # - A historical construction ("万里长城")
    texts = [
        "长城汽车发布了新款SUV并披露季度销量与营收增长情况。",
        "万里长城是中国古代的防御工程，分布在多省地区并具有重要历史价值。",
    ]
    vecs = emb.embed(texts)
    mat = normalize_rows(np.asarray(vecs, dtype=np.float32))

    # Use a conservative similarity threshold aligned with store defaults.
    clusters = greedy_cluster_by_centroid(mat, min_sim=0.8)
    # For two very different contexts, we expect two clusters.
    assert len(clusters) == 2
