"""Deterministic question classification helpers.

This module supports application-layer routing decisions such as:
- "computable" questions (time/amount/rate/threshold) should be backed by deterministic tool evidence.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_DATE_LIKE_RE = re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b")
_TIME_LIKE_RE = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
_NUMBER_RE = re.compile(r"[-+]?\d+(\.\d+)?")


@dataclass(frozen=True)
class ComputableQuestionSignals:
    is_computable: bool
    matched_keywords: list[str] = field(default_factory=list)
    matched_operators: list[str] = field(default_factory=list)
    has_number: bool = False
    has_date: bool = False
    has_time: bool = False

    def to_dict(self) -> dict:
        return {
            "is_computable": bool(self.is_computable),
            "matched_keywords": list(self.matched_keywords),
            "matched_operators": list(self.matched_operators),
            "has_number": bool(self.has_number),
            "has_date": bool(self.has_date),
            "has_time": bool(self.has_time),
        }


def detect_computable_question(
    question: str,
    *,
    keywords: Sequence[str] = (),
    operators: Sequence[str] = (),
    policy: Mapping[str, Any] | None = None,
) -> ComputableQuestionSignals:
    """Return deterministic signals indicating whether the question is "computable"."""

    text = (question or "").strip()
    if not text:
        return ComputableQuestionSignals(is_computable=False)

    lowered = text.lower()

    matched_keywords = _match_terms(lowered, keywords)
    matched_operators = _match_terms(lowered, operators)

    has_number = bool(_NUMBER_RE.search(text))
    has_date = bool(_DATE_LIKE_RE.search(text))
    has_time = bool(_TIME_LIKE_RE.search(text))

    cfg = dict(policy or {})
    min_strong_hits = int(cfg.get("min_strong_hits", 1))
    min_weak_hits = int(cfg.get("min_weak_hits", 1))
    require_weak_cue = bool(cfg.get("require_weak_cue", True))

    strong_hits = len(matched_keywords) + len(matched_operators) + (1 if has_date else 0)
    weak_hits = (1 if has_number else 0) + (1 if has_time else 0) + (1 if has_date else 0)

    strong_ok = strong_hits >= max(0, min_strong_hits)
    weak_ok = weak_hits >= max(0, min_weak_hits)
    is_computable = bool(strong_ok and (weak_ok if require_weak_cue else True))
    return ComputableQuestionSignals(
        is_computable=is_computable,
        matched_keywords=matched_keywords,
        matched_operators=matched_operators,
        has_number=has_number,
        has_date=has_date,
        has_time=has_time,
    )


def _match_terms(text: str, terms: Sequence[str]) -> list[str]:
    if not text or not terms:
        return []
    seen: set[str] = set()
    hits: list[str] = []
    for raw in terms:
        term = str(raw or "").strip()
        if not term:
            continue
        if term.lower() in text:
            token = term
            if token.lower() in seen:
                continue
            seen.add(token.lower())
            hits.append(token)
    return hits


def coerce_str_list(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for item in values:
        token = str(item or "").strip()
        if token:
            out.append(token)
    return out
