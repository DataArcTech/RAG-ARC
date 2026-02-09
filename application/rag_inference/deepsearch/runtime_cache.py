"""DeepSearch service runtime caches (application layer).

These caches are process-local and intentionally conservative:
- Owner-scoped (no cross-user reuse)
- Fingerprint-scoped (config/prompt/model changes auto-bust)
- Exact-match keys by default (avoid "similarity" mistakes)
"""
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from framework.cache import TTLRUCache


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_question(question: str) -> str:
    """Normalize question for cache keys (deterministic, non-semantic).

    Goal: increase cache hits for trivially equivalent user inputs (e.g., trailing punctuation)
    without attempting semantic rewrites (which can be unsafe).
    """
    text = str(question or "")
    # 1) Unicode normalize so full-width punctuation and compatibility forms become stable.
    text = unicodedata.normalize("NFKC", text)
    # 2) Collapse internal whitespace.
    text = " ".join(text.strip().split())

    # 3) Strip leading/trailing punctuation (keeps interior punctuation to avoid collisions like "A-B" vs "AB").
    def _is_punct(ch: str) -> bool:
        if not ch:
            return False
        try:
            return unicodedata.category(ch).startswith("P")
        except Exception:
            return False

    while text and _is_punct(text[0]):
        text = text[1:].lstrip()
    while text and _is_punct(text[-1]):
        text = text[:-1].rstrip()
    return text


@dataclass(frozen=True)
class CachedInitialThink:
    report_needed: bool
    report_style: str
    raw: dict[str, Any]
    plan_items: list[dict[str, Any]]
    think_notes_payloads: list[dict[str, Any]]


class DeepSearchInitialThinkCache:
    """Cache for the initial think result (plan + next actions)."""

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        self._cache: TTLRUCache[str, CachedInitialThink] = TTLRUCache(max_entries=max_entries, ttl_seconds=ttl_seconds)

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        return self._cache.stats()

    def make_key(
        self,
        *,
        owner_scope_id: str,
        question: str,
        service_fingerprint: str,
        llm_fingerprint: str,
        think_prompt_fingerprint: str,
    ) -> str:
        normalized = normalize_question(question)
        payload = "|".join(
            [
                "think_init",
                str(owner_scope_id),
                str(service_fingerprint),
                str(llm_fingerprint),
                str(think_prompt_fingerprint),
                _stable_hash(normalized),
            ]
        )
        return payload

    def get(self, key: str) -> Optional[CachedInitialThink]:
        return self._cache.get(key)

    def set(self, key: str, value: CachedInitialThink) -> None:
        self._cache.set(key, value)
