"""Config defaults for DeepSearch report writing.

User constraint: keep runtime behavior configurable; avoid non-config hardcoding.
"""
from dataclasses import dataclass
from typing import Optional


DEFAULT_REPORT_TEMPERATURE = 0.2
DEFAULT_REPORT_MAX_RETRIES = 2
DEFAULT_REPORT_JSON_REPAIR_ATTEMPTS = 2

DEFAULT_REPORT_MAX_EVIDENCE_ITEMS: Optional[int] = 24
DEFAULT_REPORT_MAX_EVIDENCE_CHARS = 900
DEFAULT_REPORT_MAX_GRAPH_CHAIN_ITEMS: Optional[int] = 48

DEFAULT_REPORT_PARALLEL_SECTIONS = False
DEFAULT_REPORT_MAX_PARALLEL_SECTIONS = 4
DEFAULT_REPORT_MAX_SECTION_EVIDENCE_ITEMS = 10
DEFAULT_REPORT_SYNTHESIS_SECTION_MAX_CHARS = 1200

DEFAULT_PARALLEL_TITLE_MAX_CHARS = 80
DEFAULT_ERROR_SNIPPET_LIMIT_CHARS = 500

SECTIONWISE_PREVIOUS_TITLES_MAX = 2
SECTIONWISE_PRIMARY_EVIDENCE_IDS_MAX = 12
SECTIONWISE_RETAIN_K_DEFAULT = 5


# Slimming defaults used for prompt budgeting. These are intentionally kept in config.
SLIM_METHOD_LEVEL_1_PLAN_STEPS = 12
SLIM_METHOD_LEVEL_1_REASONING_STEPS = 24
SLIM_METHOD_LEVEL_1_TOOL_RESULTS = 10

SLIM_METHOD_LEVEL_2_PLAN_STEPS = 10
SLIM_METHOD_LEVEL_2_REASONING_STEPS = 16
SLIM_METHOD_LEVEL_2_TOOL_RESULTS = 6

SLIM_GRAPH_EVIDENCE_LEVEL_SEED_ENTITIES = 12

SLIM_COVERAGE_BASE_PENDING_EXTERNAL_MAX_ITEMS = 6


@dataclass(frozen=True, slots=True)
class ReportWritePromptBudgetLevel:
    """A shrink profile for the report-writing prompt when context is exceeded.

    Divisors are applied to per-writer maxima (e.g., max_evidence_chars).
    """

    methodology_level: int
    graph_evidence_level: int
    coverage_level: int

    highlights_max_items: Optional[int]
    highlight_text_max_chars: Optional[int]

    evidence_items_divisor: int
    evidence_chars_divisor: int
    graph_chain_items_divisor: int


REPORT_WRITE_PROMPT_BUDGET_LEVELS: tuple[ReportWritePromptBudgetLevel, ...] = (
    ReportWritePromptBudgetLevel(
        methodology_level=0,
        graph_evidence_level=0,
        coverage_level=0,
        highlights_max_items=None,
        highlight_text_max_chars=None,
        evidence_items_divisor=1,
        evidence_chars_divisor=1,
        graph_chain_items_divisor=1,
    ),
    ReportWritePromptBudgetLevel(
        methodology_level=1,
        graph_evidence_level=1,
        coverage_level=1,
        highlights_max_items=4,
        highlight_text_max_chars=2000,
        evidence_items_divisor=4,
        evidence_chars_divisor=4,
        graph_chain_items_divisor=4,
    ),
    ReportWritePromptBudgetLevel(
        methodology_level=2,
        graph_evidence_level=2,
        coverage_level=2,
        highlights_max_items=2,
        highlight_text_max_chars=1200,
        evidence_items_divisor=8,
        evidence_chars_divisor=8,
        graph_chain_items_divisor=8,
    ),
    ReportWritePromptBudgetLevel(
        methodology_level=3,
        graph_evidence_level=3,
        coverage_level=3,
        highlights_max_items=1,
        highlight_text_max_chars=800,
        evidence_items_divisor=16,
        evidence_chars_divisor=16,
        graph_chain_items_divisor=16,
    ),
)
