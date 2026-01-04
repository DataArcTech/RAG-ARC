"""Centralized runtime user-facing messages for DeepSearch.

These are NOT LLM prompts; keep them centralized to avoid scattering hard-coded
strings across application/business code.
"""

from typing import Final


GENERATION_MODE_MISSING_DETERMINISTIC_TOOLS: Final[str] = "deterministic_missing_deterministic_tools"

COMPUTABLE_HARD_GATE_MESSAGE: Final[str] = (
    "This question looks computable (numbers/dates/thresholds), but the run produced no deterministic tool evidence. "
    "To avoid returning an unverifiable numeric/time answer, execution was stopped. "
    "Enable deterministic routing (probe/lead tools) or refine the query with explicit entities/predicates so "
    "deterministic tools like graph.latest_truth / graph.aggregate can be applied."
)

