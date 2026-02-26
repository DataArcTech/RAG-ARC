"""Default parameters for DeepSearch tools.

These defaults live in `config/` on purpose: core/tool code should not hard-code
search parameters so they can be tuned via configuration and regression runs.
"""

# -----------------------------
# web.search defaults
# -----------------------------
WEB_SEARCH_DEFAULT_MAX_RESULTS = 5
WEB_SEARCH_MAX_RESULTS = 10
WEB_SEARCH_DEFAULT_TIMEOUT_SECONDS = 20.0
WEB_SEARCH_DEFAULT_SEARCH_DEPTH = "advanced"
WEB_SEARCH_SNIPPET_MAX_CHARS = 900
WEB_SEARCH_AGGREGATE_ENABLED = True
WEB_SEARCH_AGGREGATE_GROUP_BY = "domain"
WEB_SEARCH_AGGREGATE_MAX_GROUPS = 3
WEB_SEARCH_AGGREGATE_MAX_RESULTS_PER_GROUP = 2

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
THINK_JSON_REPAIR_DEFAULT_ATTEMPTS = 2
THINK_JSON_REPAIR_DEFAULT_TEMPERATURE = 0.0
THINK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS = 2000
# Think responses are control/navigation JSON; keep them compact to reduce tail latency/timeouts.
# 0/None means "connector default".
THINK_DEFAULT_MAX_TOKENS = 1200
# The think prompt payload already includes structured fields (available_tools, recent_tool_runs, etc).
# Including the entire `extra` dict is redundant and can bloat prompts significantly.
THINK_INCLUDE_EXTRA_IN_PROMPT = False
# Evidence cards can grow unbounded in long runs. Keep a small, recent window in the think prompt
# and provide a deterministic L0 digest for the full evidence bank instead (OpenViking-style L0/L1/L2).
#
# 0 means "no truncation" (not recommended for long-doc runs).
THINK_CONTEXT_EVIDENCE_MAX_CARDS = 64
THINK_EVIDENCE_L0_DIGEST_ENABLED = True
THINK_EVIDENCE_L0_DIGEST_MAX_FILES = 6
THINK_EVIDENCE_L0_DIGEST_MAX_RANGES_PER_FILE = 6
THINK_CURRENT_PLAN_MAX_ITEMS = 16

# Prompt-variant + budget-status policies for think orchestration.
THINK_PROMPT_VARIANTS_ENABLED = True
THINK_BUDGET_STATUS_ENABLED = True
# Phase classification thresholds (used only to guide tool selection; does NOT bypass evidence gates).
THINK_BUDGET_LOW_REMAINING_CALLS = 3
THINK_BUDGET_CRITICAL_REMAINING_CALLS = 1
THINK_BUDGET_LOW_REMAINING_RATIO = 0.15
THINK_BUDGET_CRITICAL_REMAINING_RATIO = 0.05

# -----------------------------
# file-scope propagation policy
# -----------------------------
# DeepSearch uses a "file_scope" hint (graph_context.metadata.file_scope) to keep non-global tools
# from drifting across unrelated documents.
#
# "Global tools" are explicitly allowed to ignore file_scope so they can be used to *discover* the
# right files (routing) or search the web.
DEEPSEARCH_GLOBAL_ACTION_TOOL_NAMES = (
    "locate",
    "web.search",
)
DEEPSEARCH_GLOBAL_ACTION_TOOL_PREFIXES = ()

# -----------------------------
# explore defaults
# -----------------------------
EXPLORE_READ_MAX_CHUNKS = 12
EXPLORE_READ_MAX_CHARS = 6000

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
SEARCH_ENTITY_EXTRACT_MAX_TOKENS = 480
SEARCH_ENTITY_EXTRACT_JSON_REPAIR_ATTEMPTS = 4

# -----------------------------
# locate defaults (relevant files routing)
# -----------------------------
# `locate` is a *routing* tool: it searches globally for relevant chunks, aggregates them by file_id,
# and returns candidate files + brief "why relevant" snippets.
FILE_SEARCH_DEFAULT_TOP_K = 5
# Retrieval depth per channel (faiss/bm25/graph_chunk). Higher values improve recall but cost more.
FILE_SEARCH_CHANNEL_TOP_K = 25
# Rank fusion (RRF) constant used to aggregate chunk ranks into a per-file score.
FILE_SEARCH_RRF_K = 60
# LLM rerank defaults (intent alignment) -- kept for backward-compat / normal RAG
FILE_SEARCH_ENABLE_LLM_RERANK_DEFAULT = True
# API rerank defaults (DashScope qwen3-rerank; used by DeepSearch locate)
FILE_SEARCH_ENABLE_API_RERANK_DEFAULT = True
FILE_SEARCH_RERANK_TOP_K = 10
FILE_SEARCH_RERANK_TEMPERATURE = 0.1
# Rerank skip rule: when the top candidate score is confidently above the runner-up,
# we can skip the rerank LLM call to reduce latency/cost.
#
# IMPORTANT: this MUST be gated by query intent cues; for entity-specific existence/attribute questions
# (e.g., "Does A have X?"), skipping rerank can route to "B has X" documents incorrectly.
FILE_SEARCH_RERANK_SKIP_SCORE_MARGIN_RATIO = 0.3
FILE_SEARCH_RERANK_SKIP_BLOCK_QUERY_CUES = (
    "是否",
    "有无",
    "有没有",
    "是否有",
    "是否支持",
    "是否包含",
    "是否提供",
    "does ",
    "doesn't",
    "does not",
    "do ",
    "is there",
    "have ",
    "has ",
    "support",
    "include",
)
# How many representative hits/snippets to show per candidate file in the human-readable summary.
FILE_SEARCH_MAX_SNIPPETS_PER_FILE = 3
# Hard cap for each snippet shown in the summary (diagnostics keep the full snippet returned by channel tool).
FILE_SEARCH_SUMMARY_SNIPPET_PREVIEW_CHARS = 260

# Structure channel (page-level locate): tree propagation weight.
# After RRF fusion, boost non-hit pages in sections that contain hit pages.
# PropRAG uses PPR damping 0.45-0.75; HippoRAG uses 0.5.  Our propagation is
# simpler (linear boost, not PPR), so we use a lower value.
STRUCTURE_TREE_PROPAGATION_WEIGHT = 0.35

# -----------------------------
# page-level locate defaults
# -----------------------------
# When `locate(file=X)` is called, these override the file-level defaults.
# Research consensus (ColPali, HippoRAG, RankRAG): page-level top_k = 5-10.
PAGE_LOCATE_DEFAULT_TOP_K = 10
# Broader chunk pool per channel within a single file for better page coverage.
# Research (SBERT, ZeroEntropy): 50-100 candidates for rerank pipeline.
PAGE_LOCATE_CHANNEL_TOP_K = 50
# Reranker sees more page candidates than file-level (file-level is OFF).
PAGE_LOCATE_RERANK_TOP_K = 20

# -----------------------------
# section tree/node-type hints
# -----------------------------
# Map semantic_unit_type -> node_type label (page/image/table hints).
SECTION_NODE_TYPE_MAP = {
    "table": "table",
    "image": "image",
    "math": "equation",
    # Other common semantic_unit_type values emitted by SemanticUnitChunker.
    # These are used for navigation/diagnostics only (e.g. read.pages continuity hints).
    "list": "list",
    "code": "code",
    "blockquote": "blockquote",
    "text": "paragraph",
}
SECTION_NODE_TYPE_DEFAULT = "page"
# Limit how many chunk metadata rows to scan when deriving node-type counts for sections.
SECTION_NODE_TYPE_MAX_CHUNKS = 2000
SECTION_NODE_TYPE_MAX_PER_SECTION = 160

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
# toc + structured reading defaults (PageIndex navigation)
# -----------------------------
TOC_TREE_DEFAULT_MAX_DEPTH = 2
TOC_TREE_MAX_CHUNKS_SCANNED = 5000

# -----------------------------
# read.pages continuity signals (navigation hints only; no auto-fetch)
# -----------------------------
# These knobs control when `read.pages` suggests expanding to contiguous pages to
# avoid cutting off long spans (tables, lists, dense pages). They are used for
# *suggestions only* and must never trigger automatic extra reads.
READ_PAGES_SIGNALS_ENABLED = True
READ_PAGES_SIGNALS_EXPAND_DELTA_PAGES = 1  # suggest p-1..p+1

# Absolute + relative gates. A page triggers a continuity hint if it exceeds
# at least one absolute threshold OR is unusually large vs the median of pages
# returned in the current read.pages call.
READ_PAGES_SIGNALS_LONG_PAGE_MIN_CHARS = 9000
READ_PAGES_SIGNALS_DENSE_PAGE_MIN_CHUNKS = 24
READ_PAGES_SIGNALS_MEDIAN_MULTIPLIER = 1.8

# List-heavy pages are likely to span multiple pages.
READ_PAGES_SIGNALS_LIST_MIN_CHUNKS = 6
TOC_TREE_MAX_NODES = 220

# NOTE: `read.pages` returns full pages without truncation. Page selection (which pages to read)
# is controlled by the reasoning loop / LLM, not by per-page character caps.

# -----------------------------
# search UX hints (LLM guidance)
# -----------------------------
# When search returns chunk snippets that include page metadata, we surface suggested read.pages
# calls so the model can expand context deterministically.
SEARCH_SUGGESTED_READ_MAX = 2
SEARCH_TERM_HINTS_MAX = 8
# When scoped search yields no hits, suggest reading the first few pages (often product facts / key terms).
SEARCH_EMPTY_SUGGEST_OVERVIEW_PAGE_END = 2
