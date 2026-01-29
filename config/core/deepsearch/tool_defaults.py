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
# When tools pass "messy" entity strings (aliases, abbreviations, extra tokens),
# `graph.neighbors` may return count=0. These knobs keep the fix centralized/configurable.
NEIGHBORS_ENTITY_RESOLUTION_ENABLED = True
NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT = 12
NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN = 3
NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS = 2
# Confidence gate for auto-resolving entity_name -> best match.
NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN = 0.86
NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN = 0.06

# -----------------------------
# graph.ops defaults
# -----------------------------
GRAPH_OPS_ALLOW_CUSTOM_CYPHER = True
GRAPH_OPS_REQUIRE_OWNER_FILTER = True
GRAPH_OPS_MAX_CYPHER_CHARS = 8000
GRAPH_OPS_MAX_ROWS = 200

GRAPH_OPS_PATH_EXISTS_DEFAULT_MAX_HOPS = 4
GRAPH_OPS_PATH_EXISTS_MAX_HOPS = 20

GRAPH_OPS_NEIGHBORS_DEFAULT_LIMIT = 20
GRAPH_OPS_NEIGHBORS_MAX_LIMIT = 200

GRAPH_OPS_INTERSECTION_DEFAULT_LIMIT = 12
GRAPH_OPS_INTERSECTION_MAX_LIMIT = 50

GRAPH_OPS_SET_DIFFERENCE_DEFAULT_LIMIT = 20
GRAPH_OPS_SET_DIFFERENCE_MAX_LIMIT = 200

GRAPH_OPS_AGGREGATE_DEFAULT_LIMIT = 10
GRAPH_OPS_AGGREGATE_MAX_LIMIT = 50

GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_HOPS = 3
GRAPH_OPS_REL_PATH_EXPLORE_MAX_HOPS = 5
GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_PATHS = 200
GRAPH_OPS_REL_PATH_EXPLORE_MAX_PATHS = 2000
GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_SEQUENCES = 40
GRAPH_OPS_REL_PATH_EXPLORE_MAX_SEQUENCES = 200

GRAPH_OPS_REL_PATH_GROUND_DEFAULT_MAX_PATHS = 25
GRAPH_OPS_REL_PATH_GROUND_MAX_PATHS = 200

GRAPH_OPS_FACTS_BY_TYPE_DEFAULT_LIMIT = 50
GRAPH_OPS_FACTS_BY_TYPE_MAX_LIMIT = 300

GRAPH_OPS_EXPAND_TERMS_DEFAULT_LIMIT = 25
GRAPH_OPS_EXPAND_TERMS_MAX_LIMIT = 200

GRAPH_OPS_ENTITY_CONCEPTS_DEFAULT_LIMIT = 50
GRAPH_OPS_ENTITY_CONCEPTS_MAX_LIMIT = 200

GRAPH_OPS_SCHEMA_NODES_DEFAULT_LIMIT = 50
GRAPH_OPS_SCHEMA_NODES_MAX_LIMIT = 300

GRAPH_OPS_RULE_CHECK_DEFAULT_LIMIT = 5
GRAPH_OPS_RULE_CHECK_MAX_LIMIT = 50

GRAPH_OPS_TRACE_TO_ROOT_DEFAULT_MAX_HOPS = 6
GRAPH_OPS_TRACE_TO_ROOT_MAX_HOPS = 20

GRAPH_OPS_LATEST_TRUTH_DEFAULT_LIMIT = 1

GRAPH_OPS_SDF_DEFAULT_LIMIT = 50

# -----------------------------
# web.search defaults
# -----------------------------
WEB_SEARCH_DEFAULT_MAX_RESULTS = 5
WEB_SEARCH_MAX_RESULTS = 10
WEB_SEARCH_DEFAULT_TIMEOUT_SECONDS = 20.0
WEB_SEARCH_DEFAULT_SEARCH_DEPTH = "advanced"
WEB_SEARCH_SNIPPET_MAX_CHARS = 900
GRAPH_OPS_SDF_MAX_LIMIT = 200

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
# think tool defaults
# -----------------------------
THINK_JSON_REPAIR_DEFAULT_ATTEMPTS = 1
THINK_JSON_REPAIR_DEFAULT_TEMPERATURE = 0.0
THINK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS = 2000

# -----------------------------
# logic.check defaults
# -----------------------------
LOGIC_CHECK_DEFAULT_TEMPERATURE = 0.1
LOGIC_CHECK_JSON_REPAIR_DEFAULT_ATTEMPTS = 1
LOGIC_CHECK_JSON_REPAIR_DEFAULT_TEMPERATURE = 0.0
LOGIC_CHECK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS = 2000
LOGIC_CHECK_MAX_ASSERTIONS = 12
LOGIC_CHECK_MAX_ISSUES = 8
LOGIC_CHECK_RECENT_TOOL_RUNS_MAX = 10
LOGIC_CHECK_RECENT_TOOL_RUNS_MAX_CHARS = 320
LOGIC_CHECK_EVIDENCE_ID_MAX = 200

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

# -----------------------------
# search.file defaults (relevant files routing)
# -----------------------------
# `search.file` is a *routing* tool: it searches globally for relevant chunks, aggregates them by
# file_id, and returns candidate files + brief "why relevant" snippets.
FILE_SEARCH_DEFAULT_TOP_K = 5
# Retrieval depth per channel (faiss/bm25/graph_chunk). Higher values improve recall but cost more.
FILE_SEARCH_CHANNEL_TOP_K = 25
# Rank fusion (RRF) constant used to aggregate chunk ranks into a per-file score.
FILE_SEARCH_RRF_K = 60
# How many representative hits/snippets to show per candidate file in the human-readable summary.
FILE_SEARCH_MAX_SNIPPETS_PER_FILE = 3
# Hard cap for each snippet shown in the summary (diagnostics keep the full snippet returned by channel tool).
FILE_SEARCH_SUMMARY_SNIPPET_PREVIEW_CHARS = 260

# -----------------------------
# section.search defaults (section-level routing)
# -----------------------------
SECTION_SEARCH_DEFAULT_TOP_K = 6
SECTION_SEARCH_SUMMARY_PREVIEW_CHARS = 160

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

# -----------------------------
# explore defaults
# -----------------------------
EXPLORE_MAX_CONCURRENCY = 4
EXPLORE_READ_MAX_CHUNKS = 12
EXPLORE_READ_MAX_CHARS = 6000

# -----------------------------
# toc + structured reading defaults (PageIndex navigation)
# -----------------------------
TOC_TREE_DEFAULT_MAX_DEPTH = 6
TOC_TREE_MAX_CHUNKS_SCANNED = 5000
TOC_TREE_MAX_NODES = 220

READ_SECTION_DEFAULT_MAX_CHUNKS = 80
READ_SECTION_DEFAULT_MAX_CHARS = 16000

READ_PAGES_DEFAULT_MAX_CHUNKS = 120
READ_PAGES_DEFAULT_MAX_CHARS = 20000

# Chunk-neighbor reading (without explicit NEXT_CHUNK edges).
READ_NEIGHBORS_DEFAULT_BEFORE = 3
READ_NEIGHBORS_DEFAULT_AFTER = 3
READ_NEIGHBORS_MAX_WINDOW = 40
READ_NEIGHBORS_DEFAULT_MAX_CHUNKS = 40
READ_NEIGHBORS_DEFAULT_MAX_CHARS = 12000
