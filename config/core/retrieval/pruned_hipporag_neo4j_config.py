from typing import Literal, Optional
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jRetriever


class PrunedHippoRAGNeo4jRetrievalConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Retrieval with Neo4j backend.

    This configuration uses the same retrieval algorithm as the igraph version,
    but with Neo4j as the graph database backend instead of SQLite + igraph.

    Key differences from igraph version:
    - Graph storage: Neo4j instead of SQLite
    - Graph queries: Cypher instead of SQL
    - PageRank: Extracted subgraph to igraph (same as igraph version)
    - FAISS indices: Same (facts and entities)
    """
    type: Literal["pruned_hipporag_neo4j_retrieval"] = "pruned_hipporag_neo4j_retrieval"

    # Neo4j graph store configuration
    graph_config: PrunedHippoRAGNeo4jConfig

    # Optional LLM configuration for fact reranking
    llm_config: Optional[OpenAIChatConfig] = None

    # Fact retrieval parameters
    fact_retrieval_top_k: int = Field(
        default=20,
        description="Number of top facts to retrieve from FAISS before reranking"
    )

    # LLM reranking parameters
    enable_llm_reranking: bool = Field(
        default=True,
        description="Whether to use LLM to rerank and filter retrieved facts"
    )
    max_facts_after_reranking: int = Field(
        default=5,
        description="Maximum number of facts to keep after LLM reranking"
    )

    # Graph expansion parameters
    expansion_hops: int = Field(
        default=2,
        description="Number of hops to expand from seed entities in the graph"
    )
    include_chunk_neighbors: bool = Field(
        default=True,
        description="Whether to include chunk neighbors during graph expansion"
    )

    # Pruning parameters (query-aware)
    enable_pruning: bool = Field(
        default=True,
        description="Whether to enable query-aware pruning based on entity relevance to the query"
    )
    max_neighbors: int = Field(
        default=30,
        description="Base number of neighbors to keep per node during expansion"
    )
    query_aware_multiplier: float = Field(
        default=2.0,
        description="Multiplier for increasing max_neighbors for highly relevant entities"
    )
    query_aware_min_k: int = Field(
        default=10,
        description="Minimum number of neighbors to keep for any entity"
    )
    query_aware_max_k: int = Field(
        default=100,
        description="Maximum number of neighbors to keep for highly relevant entities"
    )

    # PageRank parameters
    damping_factor: float = Field(
        default=0.5,
        description="Damping factor for Personalized PageRank (0.5 = balanced exploration)"
    )
    passage_node_weight: float = Field(
        default=0.05,
        description="Weight assigned to passage nodes in PPR initialization"
    )

    # PPR backend selection
    ppr_backend: Literal["push", "igraph"] = Field(
        default="push",
        description="PPR computation backend: 'push' (fast, recommended) or 'igraph' (fallback)"
    )

    def build(self):
        return PrunedHippoRAGNeo4jRetriever(config=self)

