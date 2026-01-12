from types import SimpleNamespace

from core.retrieval.graph_retrieveal.graph_retrieval import GraphRetrieval
from core.retrieval.graph_retrieveal.metadata_keys import (
    GRAPH_RETRIEVAL_CHUNK_SOURCE_KEY,
    GRAPH_RETRIEVAL_FALLBACK_KEY,
)
from core.retrieval.graph_retrieveal.models import CandidateResult, ChunkScore, PPRResult, SubgraphResult


class _ExplodingGraphStore:
    def get_by_ids(self, _ids):
        raise RuntimeError("boom: get_by_ids failed")


class _DummyGraphConfig:
    type = "networkx"

    def build(self):
        return _ExplodingGraphStore()


class _DummyEmbeddingModel:
    def embed_query(self, _query):
        return [0.0]


class _DummyEmbeddingConfig:
    def build(self):
        return _DummyEmbeddingModel()


class _TestableGraphRetrieval(GraphRetrieval):
    def parallel_candidate_recall(self, query: str) -> CandidateResult:
        return CandidateResult(entity_candidates=[], chunk_candidates=[])

    def construct_subgraph(self, seed_entities, query: str) -> SubgraphResult:
        return SubgraphResult(nodes={}, edges=[], seed_entities=[])

    def compute_personalized_pagerank(self, subgraph: SubgraphResult, seed_entities):
        return PPRResult(scores={}, normalized_scores={})

    def compute_chunk_scores(self, query: str, ppr_result: PPRResult, chunk_candidates):
        return [
            ChunkScore(
                chunk_id="missing_chunk",
                content="fallback content",
                graph_score=0.1,
                embedding_score=0.2,
                final_score=0.3,
                mentioned_entities=["entity-1"],
            )
        ]

    def fusion_and_ranking(self, chunk_scores):
        return chunk_scores


def test_graph_retrieval_fallback_attaches_error_metadata() -> None:
    cfg = SimpleNamespace(
        graph_config=_DummyGraphConfig(),
        embedding_config=_DummyEmbeddingConfig(),
        llm_config=None,
        k2_entities=1,
        k1_chunks=1,
        max_hops=1,
        beam_size=1,
        damping_factor=0.85,
        max_iterations=10,
        tolerance=1e-6,
        beta1=0.7,
        beta2=0.3,
        mu1=0.3,
        mu2=0.3,
        mu3=0.4,
        gamma1=0.4,
        gamma2=0.3,
        gamma3=0.3,
        lambda1=0.6,
        lambda2=0.4,
        eta=0.2,
        top_k_entities=10,
        alpha=0.6,
        beta=0.4,
        chunks_per_entity=10,
        mention_count_boost_log_divisor=10.0,
        mention_count_boost_max=0.1,
    )
    retriever = _TestableGraphRetrieval(cfg)
    chunks = retriever.retrieve("q", top_k=1)
    assert len(chunks) == 1
    meta = chunks[0].metadata or {}
    assert meta.get(GRAPH_RETRIEVAL_CHUNK_SOURCE_KEY) == "fallback_error"
    fb = meta.get(GRAPH_RETRIEVAL_FALLBACK_KEY) or {}
    assert fb.get("reason") == "graph_store_exception"
    assert fb.get("exception_type") == "RuntimeError"

