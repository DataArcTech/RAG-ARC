from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import (
    _PrunedHippoRAGNeo4jEmbeddingsMixin,
)


def test_faiss_distance_to_similarity_cosine_ip_is_identity() -> None:
    # FAISS IndexFlatIP / HNSW with METRIC_INNER_PRODUCT returns a similarity (larger is better).
    assert _PrunedHippoRAGNeo4jEmbeddingsMixin._faiss_distance_to_similarity(
        distance=0.9, metric="cosine", index_type="flat"
    ) == 0.9
    assert _PrunedHippoRAGNeo4jEmbeddingsMixin._faiss_distance_to_similarity(
        distance=-0.2, metric="cosine", index_type="flat"
    ) == -0.2
    assert _PrunedHippoRAGNeo4jEmbeddingsMixin._faiss_distance_to_similarity(
        distance=0.7, metric="ip", index_type="flat"
    ) == 0.7


def test_faiss_distance_to_similarity_hnsw_cosine_is_shifted() -> None:
    # For normalized vectors, FAISS HNSW cosine scores are `-||u-v||^2`.
    # Since `||u-v||^2 = 2 - 2*cos`, we recover cosine via: `cos = 1 + distance/2`.
    assert _PrunedHippoRAGNeo4jEmbeddingsMixin._faiss_distance_to_similarity(
        distance=0.0, metric="cosine", index_type="hnsw"
    ) == 1.0
    # Example: cos=0.8 -> -||u-v||^2 = -(2 - 2*0.8) = -0.4
    assert _PrunedHippoRAGNeo4jEmbeddingsMixin._faiss_distance_to_similarity(
        distance=-0.4, metric="cosine", index_type="hnsw"
    ) == 0.8


def test_faiss_distance_to_similarity_l2_is_negated() -> None:
    # FAISS L2 returns a distance (smaller is better); we convert to a score by negating.
    assert _PrunedHippoRAGNeo4jEmbeddingsMixin._faiss_distance_to_similarity(
        distance=1.25, metric="l2", index_type="flat"
    ) == -1.25
