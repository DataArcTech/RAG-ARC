"""System-level intent context prompt for multi-turn SSE stream_chat.

This is a small, domain-agnostic guardrail:
- Make the model respect the user's *current* subject (anchors) for CORRECTION/TOPIC_SWITCH.
- Reduce drift by discouraging introducing unrelated entities.

Keep it centrally managed (no prompt strings scattered in business code).
"""
from typing import Final


_BASE: Final[str] = (
    "You are given an upstream intent classification and optional subject anchors for the user's latest message.\n"
    "Use them as constraints while answering.\n"
    "\n"
    "Rules:\n"
    "- Always prioritize the latest USER message over earlier turns.\n"
    "- If intent is CORRECTION: treat anchors as the correct subject; do NOT mention or answer about the mistaken subject.\n"
    "- If intent is TOPIC_SWITCH: ignore prior subject; answer about the new anchors only.\n"
    "- If intent is CLARIFICATION: revise/clarify/supplement your prior answer; do not introduce new unrelated entities.\n"
    "- If intent is CHITCHAT_ACK: respond briefly; do not introduce new domain facts.\n"
    "- If anchors are empty or conflict with the user's latest message, ask a clarifying question.\n"
)


def build_intent_context_system_prompt(*, intent: str | None, anchors: list[str] | None, reason: str | None) -> str:
    intent_token = str(intent or "").strip() or "RAG_REQUIRED"
    anchors_list = [str(a or "").strip() for a in (anchors or []) if str(a or "").strip()]
    anchors_text = ", ".join(anchors_list) if anchors_list else "(none)"
    reason_text = str(reason or "").strip()
    if reason_text:
        return f"{_BASE}\nintent={intent_token}\nanchors={anchors_text}\nreason={reason_text}".strip()
    return f"{_BASE}\nintent={intent_token}\nanchors={anchors_text}".strip()


__all__ = ["build_intent_context_system_prompt"]

