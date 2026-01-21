"""Helpers for parsing and shaping multi-turn history for RAG.

History in the SSE pipeline is currently passed as a newline-separated string:
  "user: ...\nassistant: ...\nuser: ..."

We keep parsing deterministic and conservative (ignore malformed lines).
"""
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    role: str
    content: str


def parse_role_prefixed_history(history_text: str | None) -> list[HistoryMessage]:
    if not history_text:
        return []
    out: list[HistoryMessage] = []
    for raw in str(history_text).splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        role, content = line.split(":", 1)
        role = role.strip()
        content = content.strip()
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        out.append(HistoryMessage(role=role, content=content))
    return out


def build_history_text(
    messages: Iterable[HistoryMessage],
    *,
    user_only: bool,
    most_recent_first: bool,
    max_chars: int,
) -> str | None:
    """Build a compact history_text string suitable for query rewrite prompts."""
    items = [m for m in messages if (m.role == "user" or not user_only)]
    if most_recent_first:
        items = list(reversed(items))
    lines = [f"{m.role}: {m.content}" for m in items if m.content]
    if not lines:
        return None
    text = "\n".join(lines)
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:]
    return text or None


__all__ = ["HistoryMessage", "parse_role_prefixed_history", "build_history_text"]
