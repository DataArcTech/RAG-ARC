"""Default parameters for DeepSearch tools.

These defaults live in `config/` on purpose: core/tool code should not hard-code
search parameters so they can be tuned via configuration and regression runs.
"""

BEAM_SEARCH_DEFAULT_BEAM_SIZE = 3
BEAM_SEARCH_DEFAULT_MAX_DEPTH = 3
BEAM_SEARCH_DEFAULT_TEMPERATURE = 0.2
BEAM_SEARCH_DEFAULT_PATTERN_PROBE_MAX_TERMS = 6

CHUNK_SCAN_DEFAULT_MAX_CHUNKS = 5
CHUNK_SCAN_DEFAULT_QUERY_MAX_CHARS = 240

PATTERN_PROBE_DEFAULT_MAX_TERMS = 4
PATTERN_PROBE_DEFAULT_MIN_LATIN_LENGTH = 4
PATTERN_PROBE_DEFAULT_MIN_CJK_LENGTH = 2

CONTEXT_ROLLUP_DEFAULT_WINDOW_SIZE = 6
CONTEXT_ROLLUP_DEFAULT_SNIPPET_CHARS = 400
CONTEXT_ROLLUP_DEFAULT_TEMPERATURE = 0.0

PARALLEL_THINK_DEFAULT_BRANCHES = 3
PARALLEL_THINK_DEFAULT_TEMPERATURE = 0.4
PARALLEL_THINK_DEFAULT_CONTEXT_PREVIEW_CHARS = 200
PARALLEL_THINK_DEFAULT_CONTEXT_PREVIEW_ITEMS = 3
PARALLEL_THINK_DEFAULT_CONFIDENCE_DELTA_PER_BRANCH = 0.1
PARALLEL_THINK_DEFAULT_COVERAGE_DELTA_PER_BRANCH = 0.05

HYBRID_NEIGHBORHOOD_DEFAULT_MAX_CHUNKS = 5
HYBRID_NEIGHBORHOOD_DEFAULT_PATTERN_MAX_TERMS = 3
HYBRID_NEIGHBORHOOD_DEFAULT_TRAVERSAL_STRATEGY = "ppr_chain"
HYBRID_NEIGHBORHOOD_DEFAULT_TRAVERSAL_MAX_DEPTH = 2
HYBRID_NEIGHBORHOOD_DEFAULT_SNIPPET_CHARS = 400
HYBRID_NEIGHBORHOOD_DEFAULT_SUMMARY_TEMPERATURE = 0.1

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
# graph.evidence_crosscheck defaults
# -----------------------------
EVIDENCE_CROSSCHECK_GRAPH_BACKFILL_ENABLED = True
EVIDENCE_CROSSCHECK_GRAPH_BACKFILL_MAX_CHUNKS = 8

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
