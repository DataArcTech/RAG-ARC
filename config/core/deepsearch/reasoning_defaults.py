"""Default parameters for DeepSearch reasoning UX.

These defaults live under `config/` so core orchestration code does not hard-code
token/trace shaping constants. They can be tuned via configuration/regressions.
"""

TRACE_REFLECTION_DEFAULT_EVIDENCE_SAMPLE_COUNT = 4
TRACE_REFLECTION_DEFAULT_NEW_EVIDENCE_ID_COUNT = 12
TRACE_REFLECTION_DEFAULT_MAX_LINES = 10
TRACE_REFLECTION_DEFAULT_TEMPERATURE = 0.2

# Think tool catalogs are truncated for prompt budgeting; always include a small set of
# high-value tools even when the catalog is long.
#
# Rationale: in long-document DeepSearch, `locate` + `toc.tree` + `read.pages`
# are the default navigation backbone. If these hints are truncated away, models tend to fall back
# to snippet-only guessing.
THINK_TOOL_CATALOG_ALWAYS_INCLUDE: tuple[str, ...] = (
    "locate",
    "read.pages",
    "code.python",
)

# Recent tool runs forwarded to think checkpoints (metadata/envelopes only; no truncation).
THINK_RECENT_TOOL_RUNS_MAX = 3
