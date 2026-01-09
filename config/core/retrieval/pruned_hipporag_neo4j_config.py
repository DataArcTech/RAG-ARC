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
    similarity_edge_relation: str = Field(
        default="SIMILAR_TO",
        description="Relation name used for synonymy/similarity edges.",
    )
    similarity_edge_max_hops: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Allow similarity edges only for the first N hops (0 disables).",
    )
    similarity_edge_min_similarity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum similarity required to traverse a similarity edge.",
    )
    similarity_edge_max_per_node: int = Field(
        default=20,
        ge=0,
        le=200,
        description="Max similarity edges per node per hop (0 disables traversal).",
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

    entity_chunk_count_penalty_gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=4.0,
        description="Exponent for dividing entity reset weights by entity->chunk count (1.0 matches legacy).",
    )

    # PPR backend selection
    ppr_backend: Literal["push", "igraph"] = Field(
        default="push",
        description="PPR computation backend: 'push' (fast, recommended) or 'igraph' (fallback)"
    )

    ppr_push_epsilon: float = Field(
        default=0.000001,
        gt=0,
        lt=1,
        description="Convergence epsilon for push-based PPR (smaller = more precise, slower).",
    )

    ppr_push_threshold_mode: Literal["residual", "residual_over_degree", "residual_over_weighted_degree"] = Field(
        default="residual",
        description=(
            "Push termination thresholding strategy for approximate PPR. "
            "'residual' matches legacy; degree-normalized modes reduce hub bias by requiring larger residual on high-degree nodes."
        ),
    )

    ppr_push_target_degree_penalty_gamma: float = Field(
        default=0.5,
        ge=0.0,
        le=4.0,
        description="Target-degree penalty exponent for push-based PPR transitions (0 disables).",
    )

    ppr_directed_mode: Literal["off", "auto", "on"] = Field(
        default="auto",
        description=(
            "Whether to run direction-aware PPR for direction-sensitive predicates. "
            "'auto' enables directed PPR when kg_schema declares direction_sensitive_relations; "
            "'on' forces directed PPR; 'off' keeps legacy undirected behaviour."
        ),
    )

    fact_groundability_enabled: bool = Field(
        default=True,
        description="Whether to use provenance-groundability to filter/penalize retrieved facts.",
    )
    fact_groundability_mode: Literal["hard_filter", "soft_penalty"] = Field(
        default="soft_penalty",
        description="Groundability behavior: drop ungrounded facts vs downweight them.",
    )
    fact_groundability_dense_top_k: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Top-N dense chunks used to compute provenance overlap for fact groundability.",
    )
    fact_groundability_min_overlap_count: int = Field(
        default=1,
        ge=0,
        le=50,
        description="Hard-filter threshold: require at least this many overlapping provenance chunks.",
    )
    fact_groundability_min_overlap_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Hard-filter threshold: require provenance overlap ratio >= this value.",
    )
    fact_groundability_soft_min_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Soft-penalty floor weight for facts with zero/low provenance overlap.",
    )
    fact_groundability_soft_gamma: float = Field(
        default=1.0,
        gt=0.0,
        le=8.0,
        description="Soft-penalty exponent applied to overlap ratio.",
    )
    fact_groundability_keep_missing_provenance: bool = Field(
        default=True,
        description="Whether to keep facts with missing provenance (source_chunk_ids).",
    )
    fact_groundability_missing_provenance_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Score multiplier for facts with missing provenance (kept when allowed).",
    )

    def build(self):
        return PrunedHippoRAGNeo4jRetriever(config=self)
