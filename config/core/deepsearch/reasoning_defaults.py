"""Default parameters for DeepSearch reasoning UX.

These defaults live under `config/` so core orchestration code does not hard-code
token/trace shaping constants. They can be tuned via configuration/regressions.
"""

TRACE_REFLECTION_DEFAULT_EVIDENCE_SAMPLE_COUNT = 4
TRACE_REFLECTION_DEFAULT_EVIDENCE_PREVIEW_CHARS = 220
TRACE_REFLECTION_DEFAULT_NEW_EVIDENCE_ID_COUNT = 12
TRACE_REFLECTION_DEFAULT_MAX_LINES = 10
TRACE_REFLECTION_DEFAULT_TEMPERATURE = 0.2

# Think tool catalogs are truncated for prompt budgeting; always include a minimal set of
# high-value deterministic tools even when the catalog is long.
THINK_TOOL_CATALOG_ALWAYS_INCLUDE: tuple[str, ...] = ("explore", "code.python")

# Recent tool run summaries forwarded to think checkpoints.
THINK_RECENT_TOOL_RUNS_MAX = 3
THINK_RECENT_TOOL_RUN_SUMMARY_MAX_CHARS = 320
