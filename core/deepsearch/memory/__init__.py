"""Evidence memory utilities for DeepSearch.

Introduces a Memory Bank style interface
so the writer can retrieve only the evidence needed for each section instead of
conditioning on the entire evidence pile.
"""

from .evidence_bank import EvidenceBank, EvidenceRecord
from .plan_state import PlanState, update_plan_from_think_notes

__all__ = ["EvidenceBank", "EvidenceRecord", "PlanState", "update_plan_from_think_notes"]
