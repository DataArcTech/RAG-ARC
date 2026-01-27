"""System prompts driven by upstream *routing intents*.

Keep prompts centrally managed: routing logic lives in application/core layers,
but prompt strings should not be scattered across business code.
"""

from typing import Final


_CLARIFY_REQUIRED: Final[str] = (
    "The user's message lacks key information needed to answer.\n"
    "Ask 1-3 concise clarifying questions first.\n"
    "Do NOT attempt to answer until the missing information is provided.\n"
    "Keep the response short and action-oriented.\n"
)

_ANSWER_DISSATISFIED: Final[str] = (
    "The user is dissatisfied with the previous answer.\n"
    "Respond briefly and constructively:\n"
    "- Apologize once.\n"
    "- Ask 1-3 targeted questions to locate what was wrong or missing (e.g. which point is incorrect, desired level of detail, need citations).\n"
    "- Do NOT restart retrieval or produce a long rewritten answer in this turn.\n"
)

def build_routing_intent_system_prompt(intent: str | None) -> str | None:
    token = str(intent or "").strip().upper()
    if token == "CLARIFY_REQUIRED":
        return _CLARIFY_REQUIRED
    if token == "ANSWER_DISSATISFIED":
        return _ANSWER_DISSATISFIED
    return None


__all__ = ["build_routing_intent_system_prompt"]
