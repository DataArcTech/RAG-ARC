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
    "deterministic tools like graph.ops templates (latest_truth / aggregate) can be applied."
)

MISSING_PRIMARY_EVIDENCE_HARD_GATE_MESSAGE: Final[str] = (
    "DeepSearch report generation requires at least one successful read.pages call (full-page evidence). "
    "This run produced no citeable page evidence. "
    "Next step: use explore to route to the right file (locate), then navigate (toc.tree / tree.* / section.select), "
    "and/or graph tools (graph.ops / graph.beam_search / graph.llm_chain_explorer), "
    "and finally read the relevant pages via read.pages. "
    "Search snippets are navigation-only and cannot be used as evidence. "
    "After extracting numbers/terms from read.pages, you may use code.python for deterministic verification."
)
