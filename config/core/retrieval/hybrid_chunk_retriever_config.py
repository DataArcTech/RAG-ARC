from typing import Literal

from pydantic import Field

from framework.config import AbstractConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from config.core.retrieval.graph_neighborhood_config import GraphNeighborhoodRetrieverConfig
from core.retrieval.hybrid.hybrid_chunk_retriever import HybridChunkRetriever


class HybridChunkRetrieverConfig(AbstractConfig):
    type: Literal["hybrid_chunk_retriever"] = "hybrid_chunk_retriever"

    bm25_config: TantivyBM25RetrieverConfig = Field(description="BM25 retriever config.")
    graph_config: GraphNeighborhoodRetrieverConfig = Field(description="Graph neighborhood retriever config.")

    bm25_overfetch_k: int = Field(default=60, ge=1, description="BM25 candidate budget (overfetch).")
    graph_overfetch_k: int = Field(default=60, ge=1, description="Graph candidate budget (overfetch).")

    rrf_k: int = Field(default=60, ge=1, description="RRF constant; larger reduces rank sensitivity.")
    bm25_weight: float = Field(default=1.0, ge=0.0)
    graph_weight: float = Field(default=1.0, ge=0.0)
    both_sources_bonus: float = Field(
        default=0.0,
        ge=0.0,
        description="Extra score bonus when a candidate appears in >=2 sources (precision preference).",
    )

    def build(self):
        return HybridChunkRetriever(self)
