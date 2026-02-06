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

