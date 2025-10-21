"""
Configuration for Pruned HippoRAG Retrieval

This uses the graph store with:
- FAISS Flat for fact retrieval
- FAISS HNSW for entity synonymy edges
- numpy arrays for chunk embeddings
"""

from typing import Literal, Optional
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import PrunedHippoRAGIGraphConfig
from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever


class PrunedHippoRAGRetrievalConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Retrieval
    
    This retrieval method uses the graph store with:
    - FAISS Flat for fact retrieval (exact search)
    - FAISS HNSW for entity synonymy edges (approximate search)
    - numpy arrays for chunk embeddings (brute-force search)
    
    Retrieval pipeline:
    1. Fact retrieval using FAISS Flat (exact search for all fact scores)
    2. LLM filtering to extract seed entities
    3. Subgraph expansion from seed entities
    4. PPR on subgraph with chunk similarity (numpy brute-force)
    
    Note: Embedding model is taken from graph_config.embedding
    """

    type: Literal["pruned_hipporag_retrieval"] = "pruned_hipporag_retrieval"

    # Graph config (includes embedding config)
    graph_config: PrunedHippoRAGIGraphConfig
    llm_config: Optional[AbstractConfig] = None
    
    # Fact retrieval parameters
    fact_retrieval_top_k: int = Field(
        default=20,
        description="Number of top facts to retrieve before LLM filtering"
    )
    
    # LLM reranking parameters
    enable_llm_reranking: bool = Field(
        default=True,
        description="Whether to use LLM for fact filtering"
    )
    max_facts_after_reranking: int = Field(
        default=5,
        description="Maximum number of facts to keep after LLM reranking"
    )
    
    # Subgraph expansion parameters
    expansion_hops: int = Field(
        default=2,
        description="Number of hops to expand from seed entities"
    )
    include_chunk_neighbors: bool = Field(
        default=True,
        description="Whether to include chunk neighbors during expansion"
    )

    # Pruning parameters
    enable_expansion_pruning: bool = Field(
        default=True,
        description="Whether to enable pruning during subgraph expansion"
    )
    max_neighbors: int = Field(
        default=30,
        description="Maximum number of neighbors to keep per entity during expansion"
    )

    # PPR parameters
    damping_factor: float = Field(
        default=0.5,
        description="Damping factor for PageRank (0.0 to 1.0)"
    )
    passage_node_weight: float = Field(
        default=0.05,
        description="Weight for passage nodes in PPR (same as HippoRAG)"
    )
    
    def build(self):
        """Build the retrieval system"""
        return PrunedHippoRAGRetriever(config=self)

