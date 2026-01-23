"""LLM prompts for classifying computable (numeric/temporal/rule) questions."""

from typing import Final


COMPUTABLE_CLASSIFIER_SYSTEM_PROMPT_V1_EN: Final[str] = (
    "You are a strict classifier that decides whether a user question is 'computable'.\n"
    "A computable question requires deterministic handling because it depends on numeric amounts, dates, validity windows,\n"
    "thresholds, comparisons, set logic (inclusion/exclusion), or rule/condition evaluation.\n"
    "If the question is about definitions, narrative summaries, or open-ended explanations without numeric/time commitments,\n"
    "it is NOT computable.\n"
    "Return ONLY JSON."
)

COMPUTABLE_CLASSIFIER_USER_PROMPT_V1_EN: Final[str] = (
    "Classify the following question.\n"
    "Question:\n{question}\n\n"
    "Output JSON object with keys:\n"
    "- is_computable: boolean\n"
    "- reasons: string[] (brief)\n"
    "- suggested_tools: string[] (0-6 tool names; include deterministic graph/code tools when computable)\n"
)
