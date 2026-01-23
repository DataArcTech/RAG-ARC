"""Prompt templates used by DeepSearch runtime (non-tool orchestration)."""

from typing import Final


TRACE_REFLECTION_SYSTEM_PROMPT_TEMPLATE_EN: Final[str] = (
    "You are writing a user-visible trace reflection for a research agent.\n"
    "Write concise, action-oriented notes about what was learned from the last step and what to do next.\n"
    "Do NOT reveal private chain-of-thought. Do NOT invent facts.\n"
    "Return plain text (no JSON), at most {max_lines} lines."
)

TRACE_REFLECTION_USER_PROMPT_TEMPLATE_EN: Final[str] = (
    "Question:\n{question}\n\n"
    "Last step snapshot:\n{payload}\n\n"
    "Write the reflection now."
)
