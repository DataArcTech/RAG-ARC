"""Default parameters for DeepSearch reasoning UX.

These defaults live under `config/` so core orchestration code does not hard-code
token/trace shaping constants. They can be tuned via configuration/regressions.
"""

TRACE_REFLECTION_DEFAULT_EVIDENCE_SAMPLE_COUNT = 4
TRACE_REFLECTION_DEFAULT_EVIDENCE_PREVIEW_CHARS = 220
TRACE_REFLECTION_DEFAULT_NEW_EVIDENCE_ID_COUNT = 12
TRACE_REFLECTION_DEFAULT_MAX_LINES = 10
TRACE_REFLECTION_DEFAULT_TEMPERATURE = 0.2

