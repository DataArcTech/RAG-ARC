"""Prompt templates for DeepSearch heavy (LLM) tools.

Keep these prompts centralized so tool implementations remain prompt-agnostic and
so prompt revisions can be tracked/versioned in one place.
"""

from typing import Final


# v1 prompts -------------------------------------------------------------

BEAM_SEARCH_RERANK_SYSTEM_PROMPT_V1: Final[str] = (
    "You evaluate beam search candidates on a knowledge graph. "
    'Return JSON array [{"path_id": "...", "score": 0-1}] preferring informative paths.'
)
