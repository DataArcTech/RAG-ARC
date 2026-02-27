from typing import Any, Dict, Literal, Optional, ClassVar
import logging
import threading
import warnings
from pydantic import Field, model_validator

from framework.config import AbstractConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jRetriever
from framework.shared_module_decorator import make_hashable

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PPR Presets -- each maps to a dict of parameter overrides.
# Only parameters that differ from the Pydantic field defaults are listed.
# The user sets `preset: "balanced"` (or "fast" / "thorough") and gets a
# production-quality config with ~0 additional tuning.
# Any parameter explicitly provided in the JSON/dict takes priority over
# the preset value (explicit > preset > field default).
# ---------------------------------------------------------------------------

_PPR_PRESETS: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # BALANCED (default) -- good recall + acceptable latency.
    # Matches the current production defaults so existing deployments are
    # unaffected when users start specifying `preset: "balanced"`.
    # -----------------------------------------------------------------------
    "balanced": {
        # Core PPR
        "damping_factor": 0.3,
        "expansion_hops": 2,
        "max_neighbors": 30,
        "fact_retrieval_top_k": 10,
        "max_facts_after_reranking": 5,
        "passage_node_weight": 0.05,
        # PPR engine
        "ppr_backend": "push",
        "ppr_push_epsilon": 1e-6,
        "ppr_push_threshold_mode": "residual_over_degree",
        "ppr_push_target_degree_penalty_gamma": 0.5,
        # Dense helpers
        "dense_seed_subgraph_top_k": 30,
        "dense_seed_subgraph_entity_neighbors_k": 10,
        "dense_file_closure_enabled": True,
        "dense_file_prior_enabled": True,
        "dense_file_prior_multiplier": 2.5,
        # Entity NN seeding
        "seed_entities_from_entity_nn_enabled": True,
        "seed_entities_from_entity_nn_top_k": 10,
        "seed_entities_from_entity_nn_max_extra": 3,
    },

    # -----------------------------------------------------------------------
    # FAST -- lower latency at the cost of some recall.
    # Reduces hops, neighbor budget, dense helpers; useful for interactive
    # chat or latency-critical pipelines where graph retrieval is one of
    # several fusion channels.
    # -----------------------------------------------------------------------
    "fast": {
        # Core PPR -- fewer hops, smaller subgraph
        "damping_factor": 0.3,
        "expansion_hops": 1,
        "max_neighbors": 15,
        "fact_retrieval_top_k": 10,
        "max_facts_after_reranking": 5,
        "passage_node_weight": 0.05,
        # PPR engine -- coarser convergence
        "ppr_backend": "push",
        "ppr_push_epsilon": 1e-5,
        "ppr_push_threshold_mode": "residual_over_degree",
        "ppr_push_target_degree_penalty_gamma": 0.5,
        # Dense helpers -- minimal injection
        "dense_seed_subgraph_top_k": 10,
        "dense_seed_subgraph_entity_neighbors_k": 5,
        "dense_file_closure_enabled": False,
        "dense_file_prior_enabled": False,
        # Entity NN seeding -- reduced
        "seed_entities_from_entity_nn_enabled": True,
        "seed_entities_from_entity_nn_top_k": 5,
        "seed_entities_from_entity_nn_max_extra": 1,
    },

    # -----------------------------------------------------------------------
    # THOROUGH -- maximize recall for DeepSearch / offline analysis.
    # More hops, larger neighbor budget, aggressive dense helpers.
    # -----------------------------------------------------------------------
    "thorough": {
        # Core PPR -- more hops, wider exploration
        "damping_factor": 0.25,
        "expansion_hops": 3,
        "max_neighbors": 50,
        "fact_retrieval_top_k": 20,
        "max_facts_after_reranking": 8,
        "passage_node_weight": 0.05,
        # PPR engine -- tighter convergence
        "ppr_backend": "push",
        "ppr_push_epsilon": 1e-7,
        "ppr_push_threshold_mode": "residual_over_degree",
        "ppr_push_target_degree_penalty_gamma": 0.5,
        # Dense helpers -- aggressive injection
        "dense_seed_subgraph_top_k": 50,
        "dense_seed_subgraph_entity_neighbors_k": 15,
        "dense_file_closure_enabled": True,
        "dense_file_prior_enabled": True,
        "dense_file_prior_multiplier": 3.0,
        # Entity NN seeding -- wider
        "seed_entities_from_entity_nn_enabled": True,
        "seed_entities_from_entity_nn_top_k": 20,
        "seed_entities_from_entity_nn_max_extra": 5,
    },
}

# Public API for programmatic access to available preset names.
PPR_PRESET_NAMES = tuple(_PPR_PRESETS.keys())


class PrunedHippoRAGNeo4jRetrievalConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Retrieval with Neo4j backend.

    **Simplified usage (recommended):**

    Most users only need to set ``preset`` plus a handful of essential
    parameters.  The preset fills in sensible defaults for the 50+
    advanced parameters so you don't have to tune them individually.

    .. code-block:: json

        {
            "type": "pruned_hipporag_neo4j_retrieval",
            "preset": "balanced",
            "graph_config": { ... },
            "damping_factor": 0.3,
            "expansion_hops": 2,
            "max_neighbors": 30,
            "fact_retrieval_top_k": 10,
            "max_facts_after_reranking": 5
        }

    Available presets: ``"balanced"`` (default), ``"fast"``, ``"thorough"``.

    **Essential parameters (5-8 to tune):**

    +--------------------------+----------------------------------------------+
    | Parameter                | What it controls                             |
    +==========================+==============================================+
    | ``preset``               | Overall profile (fast/balanced/thorough)      |
    | ``damping_factor``       | PPR topic sensitivity (lower = more focused) |
    | ``expansion_hops``       | Graph exploration depth (1-3)                |
    | ``max_neighbors``        | Neighbor budget per entity per hop           |
    | ``fact_retrieval_top_k`` | Facts retrieved from FAISS before reranking  |
    | ``max_facts_after_reranking`` | Facts kept after LLM reranking          |
    | ``enable_llm_reranking`` | Whether to use LLM fact filtering            |
    | ``dense_file_prior_multiplier`` | File-level boost strength             |
    +--------------------------+----------------------------------------------+

    **Advanced parameters:**

    All 50+ fine-grained parameters are still available for expert users.
    Any parameter explicitly set in the config dict takes priority over
    the preset value.  See field docstrings for details.

    **Backward compatibility:**

    Existing configs without a ``preset`` field continue to work
    unchanged -- every field retains its original Pydantic default.
    """
    type: Literal["pruned_hipporag_neo4j_retrieval"] = "pruned_hipporag_neo4j_retrieval"

    # -----------------------------------------------------------------------
    # Preset selector -- the primary simplification lever.
    # -----------------------------------------------------------------------
    preset: Optional[Literal["balanced", "fast", "thorough"]] = Field(
        default=None,
        description=(
            "PPR tuning preset.  Sets sensible defaults for all advanced parameters. "
            "Any parameter explicitly provided in the config takes priority over the preset value. "
            "Available presets: 'balanced' (default behavior), 'fast' (lower latency), 'thorough' (max recall). "
            "When None, raw Pydantic field defaults are used (backward compatible)."
        ),
    )

    # -----------------------------------------------------------------------
    # Infrastructure (not tuning-related)
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # ESSENTIAL PARAMETERS -- the 5-8 knobs most users should tune.
    # -----------------------------------------------------------------------

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

    # Pruning parameters
    max_neighbors: int = Field(
        default=30,
        description="Base number of neighbors to keep per node during expansion"
    )

    # PageRank parameters
    damping_factor: float = Field(
        default=0.3,
        description=(
            "Damping factor for Personalized PageRank. Lower values make PPR more topic-sensitive and reduce diffusion drift "
            "(default tuned from local HippoRAG recall experiments)."
        ),
    )

    # -----------------------------------------------------------------------
    # ADVANCED PARAMETERS -- rarely need tuning; preset fills good defaults.
    # Grouped by feature area for readability.
    # -----------------------------------------------------------------------

    # -- Graph expansion (advanced) --
    include_chunk_neighbors: bool = Field(
        default=True,
        description="Whether to include chunk neighbors during graph expansion"
    )

    enable_pruning: bool = Field(
        default=True,
        description="Whether to enable query-aware pruning based on entity relevance to the query"
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

    # -- PageRank engine (advanced) --
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

    # -- Fact groundability --
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

    # -- Chunk selection strategy --
    chunk_selection_strategy: Literal["top_entity_neighbors", "top_ppr_chunks"] = Field(
        default="top_ppr_chunks",
        description=(
            "How to select the final top-k chunks from PPR results. "
            "`top_entity_neighbors` selects chunks connected to the top PPR entity (legacy, can drift across similar products). "
            "`top_ppr_chunks` returns the top PPR-ranked chunks directly (recommended for precision)."
        ),
    )

    # -- Dense seed subgraph injection --
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

    # -- Dense file closure --
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

    # -- Entity NN seeding --
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

    # -- Dense mix-in --
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

    # -- Dense file prior --
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

    # -----------------------------------------------------------------------
    # Preset application logic
    # -----------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        """
        Apply preset defaults *before* Pydantic fills in field defaults.

        Priority order (highest to lowest):
          1. Explicitly provided values in the input dict/JSON
          2. Preset values
          3. Pydantic field defaults

        This ensures backward compatibility: configs without ``preset``
        behave exactly as before.
        """
        if not isinstance(data, dict):
            return data

        preset_name = data.get("preset")
        if not preset_name:
            return data

        preset_name = str(preset_name).strip().lower()
        preset_values = _PPR_PRESETS.get(preset_name)
        if preset_values is None:
            warnings.warn(
                f"Unknown PPR preset '{preset_name}'. "
                f"Available presets: {', '.join(sorted(_PPR_PRESETS))}. "
                f"Ignoring preset and using field defaults.",
                UserWarning,
                stacklevel=2,
            )
            return data

        # Merge: preset values are used only for keys not explicitly provided.
        merged = dict(preset_values)
        merged.update(data)
        return merged

    @model_validator(mode="after")
    def _validate_cross_field_consistency(self) -> "PrunedHippoRAGNeo4jRetrievalConfig":
        """
        Emit warnings for common misconfigurations.

        These are *warnings*, not errors, to avoid breaking existing configs.
        """
        # Warn if LLM reranking is enabled but no LLM config is provided
        if self.enable_llm_reranking and self.llm_config is None:
            _logger.debug(
                "enable_llm_reranking=True but llm_config is None; "
                "LLM reranking will be silently skipped at runtime."
            )

        # Warn if dense_file_prior is enabled but dense_seed_subgraph is disabled
        if self.dense_file_prior_enabled and self.dense_seed_subgraph_top_k <= 0:
            _logger.debug(
                "dense_file_prior_enabled=True but dense_seed_subgraph_top_k=0; "
                "the file prior has limited effect without dense seed injection."
            )

        # Warn if expansion_hops is high without pruning
        if self.expansion_hops >= 3 and not self.enable_pruning:
            _logger.debug(
                "expansion_hops=%d with enable_pruning=False may cause very large subgraphs.",
                self.expansion_hops,
            )

        return self

    # -----------------------------------------------------------------------
    # Helper: describe effective config for debugging / logging
    # -----------------------------------------------------------------------

    def describe_essential(self) -> Dict[str, Any]:
        """Return a compact dict of the essential tuning parameters for logging."""
        return {
            "preset": self.preset,
            "damping_factor": self.damping_factor,
            "expansion_hops": self.expansion_hops,
            "max_neighbors": self.max_neighbors,
            "fact_retrieval_top_k": self.fact_retrieval_top_k,
            "max_facts_after_reranking": self.max_facts_after_reranking,
            "enable_llm_reranking": self.enable_llm_reranking,
            "ppr_backend": self.ppr_backend,
            "dense_file_prior_enabled": self.dense_file_prior_enabled,
            "dense_file_prior_multiplier": self.dense_file_prior_multiplier,
            "dense_file_closure_enabled": self.dense_file_closure_enabled,
            "seed_entities_from_entity_nn_enabled": self.seed_entities_from_entity_nn_enabled,
        }

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any]:
        """Return a copy of the named preset's parameter overrides."""
        name = str(name or "").strip().lower()
        preset = _PPR_PRESETS.get(name)
        if preset is None:
            raise ValueError(
                f"Unknown PPR preset '{name}'. "
                f"Available presets: {', '.join(sorted(_PPR_PRESETS))}"
            )
        return dict(preset)

    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        """Return all available presets (name -> parameter dict)."""
        return {name: dict(values) for name, values in _PPR_PRESETS.items()}

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
