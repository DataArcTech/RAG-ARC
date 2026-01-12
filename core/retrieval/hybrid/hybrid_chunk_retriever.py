import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_neighborhood_retriever import GraphNeighborhoodRetriever
from core.retrieval.hybrid.metadata_keys import (
    HYBRID_RETRIEVAL_SCORE_COMPONENTS_KEY,
    HYBRID_RETRIEVAL_SCORE_KEY,
    HYBRID_RETRIEVAL_SOURCES_KEY,
)
from core.retrieval.hybrid.rrf import RRFInput, fuse_rrf_with_scores

if TYPE_CHECKING:
    from config.core.retrieval.hybrid_chunk_retriever_config import HybridChunkRetrieverConfig

logger = logging.getLogger(__name__)


class HybridChunkRetriever:
    """Hybrid retriever for 方案一：多源候选 + 统一融合。

    Current sources (M2): BM25 + GraphNeighborhood.
    Fusion: Reciprocal Rank Fusion (RRF) for score-agnostic merging.
    """

    def __init__(self, config: "HybridChunkRetrieverConfig"):
        self.config = config
        self.bm25 = config.bm25_config.build()
        self.graph = GraphNeighborhoodRetriever(config.graph_config)

    def invoke(self, query: str, *, owner_id: str, k: int = 10) -> list[Chunk]:
        k = int(k)
        bm25_overfetch = int(self.config.bm25_overfetch_k)
        graph_overfetch = int(self.config.graph_overfetch_k)

        bm25_weight = float(getattr(self.config, "bm25_weight", 1.0) or 1.0)
        graph_weight = float(getattr(self.config, "graph_weight", 1.0) or 1.0)
        rrf_k = int(getattr(self.config, "rrf_k", 60) or 60)
        both_sources_bonus = float(getattr(self.config, "both_sources_bonus", 0.0) or 0.0)

        bm25_chunks = list(self.bm25.invoke(query, owner_id=owner_id, k=bm25_overfetch) or [])
        graph_chunks, graph_trace = self.graph.retrieve_with_trace(query, k=graph_overfetch)

        bm25_ranked = {c.id: (c, i + 1) for i, c in enumerate(bm25_chunks) if getattr(c, "id", None)}
        graph_ranked = {c.id: (c, i + 1) for i, c in enumerate(graph_chunks) if getattr(c, "id", None)}

        ranked_ids, fused_scores = fuse_rrf_with_scores(
            sources={
                "bm25": [RRFInput(id=cid, rank=rank) for cid, (_, rank) in bm25_ranked.items()],
                "graph_neighborhood": [RRFInput(id=cid, rank=rank) for cid, (_, rank) in graph_ranked.items()],
            },
            rrf_k=rrf_k,
            weights={"bm25": bm25_weight, "graph_neighborhood": graph_weight},
            both_sources_bonus=both_sources_bonus,
            k=k,
        )

        out: list[Chunk] = []
        for cid in ranked_ids:
            chosen = bm25_ranked.get(cid) or graph_ranked.get(cid)
            if chosen is None:
                continue
            chunk = chosen[0]
            meta = dict(getattr(chunk, "metadata", None) or {})

            sources: list[str] = []
            score_components: dict[str, object] = {"rrf_k": rrf_k}

            if cid in bm25_ranked:
                sources.append("bm25")
                score_components["bm25_rank"] = bm25_ranked[cid][1]
                score_components["bm25_weight"] = bm25_weight
                score_components["bm25_score"] = (bm25_ranked[cid][0].metadata or {}).get("score")
            if cid in graph_ranked:
                sources.append("graph_neighborhood")
                score_components["graph_rank"] = graph_ranked[cid][1]
                score_components["graph_weight"] = graph_weight
                score_components["graph_neighborhood_score"] = (graph_ranked[cid][0].metadata or {}).get("graph_neighborhood_score")
            if not graph_ranked:
                score_components["graph_fallback"] = graph_trace
            if both_sources_bonus > 0:
                score_components["both_sources_bonus"] = both_sources_bonus

            meta.update(
                {
                    "retrieval_path": "hybrid",
                    HYBRID_RETRIEVAL_SOURCES_KEY: sources,
                    HYBRID_RETRIEVAL_SCORE_KEY: float(fused_scores.get(cid, 0.0)),
                    HYBRID_RETRIEVAL_SCORE_COMPONENTS_KEY: score_components,
                }
            )
            chunk.metadata = meta
            out.append(chunk)

        return out
