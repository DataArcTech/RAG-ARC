from typing import Literal, Optional, ClassVar
import threading
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jRetriever
from framework.shared_module_decorator import make_hashable


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

    # Process-level cache toggle (via @shared_module on PrunedHippoRAGNeo4jRetriever).
    # Default True: identical configs share one in-process retriever instance, so expensive graph caches load once.
    shared_instance: bool = Field(
        default=True,
        description=(
            "Whether to reuse a process-level shared PrunedHippoRAGNeo4jRetriever instance for identical configs. "
            "Disable for strict isolation in tests/multi-tenant workers."
        ),
    )

    node_mappings_cache_max_entries: int = Field(
        default=2,
        ge=0,
        description=(
            "Max number of (owner_id, graph_store_cache_version) node-mapping entries to keep in the in-process cache. "
            "These mappings contain the chunk-id list and a dense passage embedding matrix used by graph retrieval. "
            "0 disables the shared cache (will rebuild per thread/request)."
        ),
    )

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
        default=0.3,
        description=(
            "Damping factor for Personalized PageRank. Lower values make PPR more topic-sensitive and reduce diffusion drift "
            "(default tuned from local HippoRAG recall experiments)."
        ),
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

    entity_reset_weight_aggregation: Literal["overwrite", "sum_avg"] = Field(
        default="sum_avg",
        description=(
            "How to aggregate multiple fact-derived weights for the same entity in the PPR reset distribution. "
            "'overwrite' keeps the last-seen fact score (legacy in this repo); "
            "'sum_avg' accumulates per-fact contributions then averages (closer to upstream HippoRAG)."
        ),
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
        default="residual_over_degree",
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
        default="off",
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

    chunk_selection_strategy: Literal["top_entity_neighbors", "top_ppr_chunks"] = Field(
        default="top_ppr_chunks",
        description=(
            "How to select the final top-k chunks from PPR results. "
            "`top_entity_neighbors` selects chunks connected to the top PPR entity (legacy, can drift across similar products). "
            "`top_ppr_chunks` returns the top PPR-ranked chunks directly (recommended for precision)."
        ),
    )

    dense_seed_subgraph_top_k: int = Field(
        default=30,
        ge=0,
        le=200,
        description=(
            "Inject the top-N dense chunk matches into the PPR subgraph node set before running PageRank. "
            "This keeps the final selection PPR-based while improving recall for queries with explicit product names "
            "that may not appear in graph facts/entities. Set to 0 to disable."
        ),
    )
    dense_seed_subgraph_entity_neighbors_k: int = Field(
        default=10,
        ge=0,
        le=200,
        description=(
            "When `dense_seed_subgraph_top_k` injects chunk nodes, also inject up to this many entity neighbors "
            "(MENTIONS/RELATES_TO-connected entity nodes) per injected chunk into the PPR subgraph. "
            "This keeps injected dense chunks connected to the entity graph and improves recall when product/entity "
            "names are present in dense hits but absent from fact-derived seeds. Set to 0 to disable."
        ),
    )

    dense_file_closure_enabled: bool = Field(
        default=True,
        description=(
            "Whether to inject chunks from the top dense-matched file into the PPR subgraph. "
            "This is domain-agnostic and improves completeness for single-document queries without switching to full-graph PPR. "
            "Gated by dense file concentration (top ratio / margin)."
        ),
    )
    dense_file_closure_top_k: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Top-N dense chunks used to compute the dominant file ratio for dense_file_closure gating.",
    )
    dense_file_closure_min_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Apply dense_file_closure only when the top file occupies at least this fraction of dense top_k.",
    )
    dense_file_closure_min_margin: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Apply dense_file_closure only when (top_ratio - second_ratio) >= this margin (reduces harm on multi-file queries).",
    )
    dense_file_closure_max_chunks: int = Field(
        default=0,
        ge=0,
        le=20000,
        description=(
            "Cap the number of chunks injected by dense_file_closure for the dominant file (0 means inject all chunks of that file)."
        ),
    )

    seed_entities_from_entity_nn_enabled: bool = Field(
        default=True,
        description=(
            "Whether to add additional seed entities from the entity FAISS nearest-neighbor index. "
            "This is domain-agnostic and helps when fact retrieval or fact->entity extraction misses relevant entities "
            "(e.g., multi-entity queries, naming variants, mixed scripts)."
        ),
    )
    seed_entities_from_entity_nn_top_k: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Top-N entity nearest neighbors to consider as candidate seed entities (0 disables).",
    )
    seed_entities_from_entity_nn_max_extra: int = Field(
        default=3,
        ge=0,
        le=100,
        description=(
            "When fact-derived seed entities are present, add at most this many additional entity-NN seeds. "
            "This keeps entity-NN seeding conservative and avoids drifting to unrelated entities."
        ),
    )
    seed_entities_from_entity_nn_max_total: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Hard cap for total seed entities after merging fact-derived seeds and entity-NN seeds (0 disables).",
    )

    dense_mix_in_top_k: int = Field(
        default=0,
        ge=0,
        le=50,
        description=(
            "Always mix in the top-N dense passage matches (by chunk embeddings) into the final chunk list. "
            "This improves precision for queries containing explicit product/file names that may not be captured "
            "by fact/entity graph signals. Set to 0 to disable."
        ),
    )

    dense_file_prior_enabled: bool = Field(
        default=True,
        description=(
            "Whether to apply an adaptive file-level prior derived from dense top-K chunks. "
            "When enabled and the dense top-K concentrates on a single file, the PPR reset weights of chunks "
            "from that file are boosted (still PPR-based), improving product-specific recall."
        ),
    )
    dense_file_prior_top_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Top-N dense chunks used to estimate the dominant file for the file prior.",
    )
    dense_file_prior_min_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum dominant-file ratio in dense top-K required to activate the file prior.",
    )
    dense_file_prior_min_margin: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum (top_ratio - second_ratio) margin required to activate the file prior. "
            "This avoids applying a single-file boost when dense evidence is split across multiple files."
        ),
    )
    dense_file_prior_max_second_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional additional gate for dense_file_prior: require second_ratio <= this threshold. "
            "When set, the prior only activates when the dense top-K is extremely concentrated on one file "
            "(i.e., there is no strong second candidate)."
        ),
    )
    dense_file_prior_max_files: int = Field(
        default=2,
        ge=1,
        le=10,
        description=(
            "Apply the dense-derived file prior to up to the top-N files (ranked by frequency in dense top-K). "
            "Secondary files are boosted with a scaled multiplier, improving multi-file coverage without losing stability."
        ),
    )
    dense_file_prior_multiplier: float = Field(
        default=2.5,
        gt=0.0,
        le=20.0,
        description="Multiplier applied to dense-derived passage reset weights for chunks in the dominant file.",
    )
    dense_file_prior_lexical_enabled: bool = Field(
        default=True,
        description=(
            "Whether to add an extra lexical gate for dense_file_prior. "
            "When enabled, the file prior only activates if the dominant file's filename-derived title "
            "is sufficiently mentioned by the query (using lightweight token overlap, no domain rules)."
        ),
    )
    dense_file_prior_lexical_min_top_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Apply the lexical gate only when the dense top-K is already fairly concentrated (top_ratio >= this value). "
            "This avoids blocking the prior on mixed-intent / multi-file queries where top_ratio is low."
        ),
    )
    dense_file_prior_lexical_min_title_coverage: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Lexical gate threshold for dense_file_prior: require that the query covers at least this fraction "
            "of the dominant file's title tokens."
        ),
    )
    dense_file_prior_lexical_min_overlap_tokens: int = Field(
        default=2,
        ge=0,
        le=50,
        description=(
            "Lexical gate threshold for dense_file_prior: require at least this many overlapping tokens between "
            "the query variants and the dominant file's title tokens."
        ),
    )
    dense_file_prior_lexical_min_margin: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Optional lexical gate margin for dense_file_prior: require (top_title_coverage - second_title_coverage) "
            ">= this value. Set to 0 to disable."
        ),
    )

    def build(self):
        if not bool(getattr(self, "shared_instance", True)):
            return PrunedHippoRAGNeo4jRetriever(config=self)
        key = make_hashable(self.model_dump())
        with self._process_cache_lock:
            cached = self._process_cache.get(key)
            if cached is not None:
                return cached
            inst = PrunedHippoRAGNeo4jRetriever(config=self)
            self._process_cache[key] = inst
            return inst

    _process_cache_lock: ClassVar[threading.Lock] = threading.Lock()
    _process_cache: ClassVar[dict[object, PrunedHippoRAGNeo4jRetriever]] = {}

    @classmethod
    def clear_process_cache(cls) -> None:
        with cls._process_cache_lock:
            cls._process_cache.clear()
