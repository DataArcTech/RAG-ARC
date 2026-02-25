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
    "explore",
    "code.python",
)

# Recent tool runs forwarded to think checkpoints (metadata/envelopes only; no truncation).
THINK_RECENT_TOOL_RUNS_MAX = 3

# Hard gate for report-style DeepSearch: before the runtime proceeds to report composition,
# we require at least one successful `read.pages` evidence. This keeps the report grounded
# in full-page context instead of snippet-only navigation.
REPORT_HARD_GATE_MIN_PRIMARY_PAGE_EVIDENCE = 1

# When the reasoning loop exits without satisfying the report hard gate, the service runtime
# may re-enter the reasoning stage with an explicit gate-failure note so the LLM can retry.
# This is a safety net for brittle model/tool interactions (JSON parse failures, early stop, etc.).
#
# Keep this value configurable (service config can override) and conservative by default.
REPORT_HARD_GATE_MAX_REASONING_RETRIES = 2
