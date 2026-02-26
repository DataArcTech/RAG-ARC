"""Default knobs for DeepSearch runtime caches.

Centralized in `config/` to avoid scattered "magic numbers" in tool/runtime code.
All caches are process-local and must remain owner-scoped to avoid future multi-tenant leaks.
"""

# Initial think (plan) cache
DEFAULT_PLAN_CACHE_ENABLED = True
DEFAULT_PLAN_CACHE_MAX_ENTRIES = 256
DEFAULT_PLAN_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

# PageIndex navigation caches (derived guidance; safe to cache briefly)
DEFAULT_TOC_TREE_CACHE_ENABLED = True
DEFAULT_TOC_TREE_CACHE_MAX_ENTRIES = 128
DEFAULT_TOC_TREE_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

DEFAULT_SECTION_NODES_CACHE_ENABLED = True
DEFAULT_SECTION_NODES_CACHE_MAX_ENTRIES = 128
DEFAULT_SECTION_NODES_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

# Run-scoped tool memoization (graph reasoning)
# - This cache is per-run (stored in ContextVar) and bounded (LRU).
# - It prevents the LLM from paying repeated identical tool invocations within one run.
DEFAULT_RUN_TOOL_MEMO_ENABLED = True
DEFAULT_RUN_TOOL_MEMO_MAX_ENTRIES = 512
# Cache only navigation/retrieval tools by default; never cache LLM think/code tools.
DEFAULT_RUN_TOOL_MEMO_INCLUDE_PREFIXES = (
    "locate",
    "toc.tree",
    "read.pages",
)
DEFAULT_RUN_TOOL_MEMO_EXCLUDE_PREFIXES = (
    "think",
    "code.python",
    "web.search",
)
