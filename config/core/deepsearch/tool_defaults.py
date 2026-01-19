"""Default parameters for DeepSearch tools.

These defaults live in `config/` on purpose: core/tool code should not hard-code
search parameters so they can be tuned via configuration and regression runs.
"""

BEAM_SEARCH_DEFAULT_BEAM_SIZE = 3
BEAM_SEARCH_DEFAULT_MAX_DEPTH = 3
BEAM_SEARCH_DEFAULT_TEMPERATURE = 0.2
BEAM_SEARCH_DEFAULT_SEED_ENTITY_TOP_K = 6
BEAM_SEARCH_SEED_EXTRACT_TEMPERATURE = 0.0
BEAM_SEARCH_SEED_EXTRACT_MAX_TOKENS = 240

# -----------------------------
# graph.neighbors defaults
# -----------------------------
# When planners pass "messy" entity strings (aliases, abbreviations, extra tokens),
# `graph.neighbors` may return count=0. These knobs keep the fix centralized/configurable.
NEIGHBORS_ENTITY_RESOLUTION_ENABLED = True
NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT = 12
NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN = 3
NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS = 2
# Confidence gate for auto-resolving entity_name -> best match.
NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN = 0.86
NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN = 0.06

# -----------------------------
# Shared graph entity resolution defaults (all graph tools)
# -----------------------------
# Online hybrid entity resolution is a shared capability across graph tools.
# Defaults are intentionally conservative to minimize mismatches/noise.
ENTITY_RESOLUTION_ENABLE_ALIAS = True
ENTITY_RESOLUTION_ENABLE_TOKEN_OVERLAP = True
# Embedding/entity-FAISS recall is enabled by default but only used when deterministic recall fails.
ENTITY_RESOLUTION_ENABLE_EMBEDDING_FALLBACK = True
ENTITY_RESOLUTION_FAISS_TOP_K = 16
ENTITY_RESOLUTION_FAISS_MIN_SIMILARITY = None  # set a float to hard-gate FAISS candidates (cosine)

# Candidate validity gate: validate edges first; fall back to chunk mentions only when needed.
ENTITY_RESOLUTION_VALIDATE_EDGES_FIRST = True
ENTITY_RESOLUTION_REQUIRE_MIN_EDGE_COUNT = 1
ENTITY_RESOLUTION_ENABLE_CHUNK_VALIDATION = True
ENTITY_RESOLUTION_REQUIRE_MIN_MENTION_COUNT = 1

# Scoring weights (normalize inside resolver).
ENTITY_RESOLUTION_SCORE_WEIGHT_TOKEN_F1 = 0.7
ENTITY_RESOLUTION_SCORE_WEIGHT_CHAR_RATIO = 0.3
ENTITY_RESOLUTION_ALIAS_SCORE_BONUS = 0.12

# -----------------------------
# graph.think tool defaults
# -----------------------------
THINK_JSON_REPAIR_DEFAULT_ATTEMPTS = 1
THINK_JSON_REPAIR_DEFAULT_TEMPERATURE = 0.0
THINK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS = 2000

# -----------------------------
# code.python tool defaults
# -----------------------------
# Keep this list small and finance-focused; expand via config when needed.
CODE_PYTHON_DEFAULT_ALLOWED_IMPORTS = (
    "json",
    "math",
    "decimal",
    "fractions",
    "statistics",
    "datetime",
    "numpy",
    "pandas",
    "scipy",
)

CODE_PYTHON_DEFAULT_TIMEOUT_SECONDS = 6.0
CODE_PYTHON_DEFAULT_MAX_CODE_CHARS = 12000
CODE_PYTHON_DEFAULT_MAX_STDOUT_CHARS = 8000
CODE_PYTHON_DEFAULT_MAX_STDERR_CHARS = 8000
CODE_PYTHON_DEFAULT_MAX_RESULT_CHARS = 12000
CODE_PYTHON_DEFAULT_MAX_MEMORY_MB = 1024
CODE_PYTHON_DEFAULT_EMIT_RESULT_EVIDENCE = True

# Internal output/diagnostics knobs (kept here to avoid scattered thresholds).
CODE_PYTHON_MIN_TIMEOUT_SECONDS = 0.1
CODE_PYTHON_SUMMARY_PREVIEW_CHARS = 240
CODE_PYTHON_BAD_RUNNER_STDOUT_PREVIEW_CHARS = 800
CODE_PYTHON_BAD_RUNNER_STDERR_PREVIEW_CHARS = 800
CODE_PYTHON_RUNNER_STDERR_PREVIEW_CHARS = 1200
CODE_PYTHON_TRACEBACK_LIMIT = 8

# -----------------------------
# search tool defaults
# -----------------------------
SEARCH_DEFAULT_CHANNELS = ("faiss", "bm25", "graph_chunk")
SEARCH_DEFAULT_TOP_K = 15
SEARCH_DEFAULT_QUERY_MAX_CHARS = 240

SEARCH_SUMMARY_MAX_TOKENS = 120
SEARCH_SUMMARY_HEAD_RATIO = 0.3
SEARCH_SUMMARY_MID_RATIO = 0.4
SEARCH_SUMMARY_TAIL_RATIO = 0.3
SEARCH_SUMMARY_SEPARATOR = " ... "

SEARCH_BM25_SNIPPET_MAX_CHARS = 320
SEARCH_BM25_HIGHLIGHT_WINDOW_CHARS = 320
SEARCH_BM25_HIGHLIGHT_PREFIX = "<<"
SEARCH_BM25_HIGHLIGHT_SUFFIX = ">>"
SEARCH_BM25_MIN_TOKEN_LENGTH = 2

SEARCH_GRAPH_USE_PPR_DEFAULT = False
SEARCH_GRAPH_ENABLE_LLM_RERANK_DEFAULT = False
SEARCH_GRAPH_ENTITY_SEED_TOP_K = 10
SEARCH_GRAPH_ENABLE_ENTITY_FALLBACK = True
SEARCH_ENTITY_EXTRACT_TEMPERATURE = 0.0
SEARCH_ENTITY_EXTRACT_MAX_TOKENS = 240

# Allowlist for per-request graph retrieval overrides supplied via tool_args.
# These map to fields in `config/core/retrieval/pruned_hipporag_neo4j_config.py`.
SEARCH_GRAPH_SAFE_OVERRIDE_KEYS = (
    "fact_retrieval_top_k",
    "max_facts_after_reranking",
    "expansion_hops",
    "include_chunk_neighbors",
    "enable_pruning",
    "max_neighbors",
    "query_aware_multiplier",
    "query_aware_min_k",
    "query_aware_max_k",
    "similarity_edge_max_hops",
    "similarity_edge_min_similarity",
    "similarity_edge_max_per_node",
    "seed_entities_from_entity_nn_enabled",
    "seed_entities_from_entity_nn_top_k",
    "seed_entities_from_entity_nn_max_extra",
    "seed_entities_from_entity_nn_max_total",
    "damping_factor",
    "ppr_backend",
    "ppr_directed_mode",
    "ppr_push_epsilon",
    "ppr_push_threshold_mode",
    "ppr_push_target_degree_penalty_gamma",
    "entity_reset_weight_aggregation",
    "entity_chunk_count_penalty_gamma",
    "dense_seed_subgraph_top_k",
    "dense_seed_subgraph_entity_neighbors_k",
    "dense_mix_in_top_k",
    "chunk_selection_strategy",
    "fact_groundability_enabled",
    "fact_groundability_mode",
    "fact_groundability_dense_top_k",
    "fact_groundability_min_overlap_count",
    "fact_groundability_min_overlap_ratio",
    "fact_groundability_soft_min_weight",
    "fact_groundability_soft_gamma",
    "fact_groundability_keep_missing_provenance",
    "fact_groundability_missing_provenance_weight",
)
