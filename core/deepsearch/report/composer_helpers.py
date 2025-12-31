"""Shared helper functions for DeepSearch report composition.

This module hosts small, testable helpers used by `core.deepsearch.report.composer`.
"""

import re
from typing import Any, Dict, List

from core.utils.stopwords import get_stopwords

from config.core.deepsearch import report_composer_defaults as composer_defaults
from config.core.deepsearch.stopwords import EVIDENCE_RANK_STOPWORDS
from core.deepsearch.utils.file_scope import normalize_filename_token

_EN_STOPWORDS = frozenset(word.lower() for word in get_stopwords("en"))


def _rank_evidences_for_question(question: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministically rank evidences for prompt inclusion (match-first, score-second)."""

    if not items:
        return []
    q = str(question or "")
    anchors: list[str] = []
    # Language-agnostic anchor extraction: pull compact unicode word tokens + numbers.
    for token in re.findall(r"\d[\d,\.%]*", q, flags=re.UNICODE):
        token = token.strip()
        if not token:
            continue
        if token not in anchors:
            anchors.append(token)
    token_re = re.compile(
        rf"[^\W_]{{{composer_defaults.DEFAULT_EVIDENCE_RANK_ASCII_ANCHOR_MIN},{composer_defaults.DEFAULT_EVIDENCE_RANK_ASCII_ANCHOR_MAX}}}",
        flags=re.UNICODE,
    )
    for token in token_re.findall(q):
        token = token.strip()
        if not token or token in EVIDENCE_RANK_STOPWORDS:
            continue
        if _EN_STOPWORDS and token.lower() in _EN_STOPWORDS:
            continue
        if token not in anchors:
            anchors.append(token)

    normalized_anchors = [a for a in anchors if a and a not in EVIDENCE_RANK_STOPWORDS][
        : composer_defaults.DEFAULT_EVIDENCE_RANK_MAX_ANCHORS
    ]

    def _score(ev: Dict[str, Any]) -> tuple[int, int, float, int, str]:
        content = str(ev.get("content") or "")
        lowered = content.lower()
        match_count = 0
        for term in normalized_anchors:
            term_lower = term.lower()
            if term_lower and term_lower in lowered:
                match_count += 1
        raw_score = ev.get("score")
        try:
            numeric_score = float(raw_score) if raw_score is not None else 0.0
        except Exception:
            numeric_score = 0.0
        chunk_id = str(ev.get("chunk_id") or "")
        source = str(ev.get("source") or "").strip().lower()
        toolish_id = ":" in chunk_id or any(chunk_id.startswith(prefix) for prefix in composer_defaults.TOOLISH_CHUNK_ID_PREFIXES)
        toolish_source = source in composer_defaults.TOOLISH_SOURCE_NAMES
        primary = 0 if (toolish_id or toolish_source) else 1
        # Prefer matched content, then score, then shorter snippets (for prompt efficiency).
        return (primary, 1 if match_count > 0 else 0, match_count, numeric_score, -len(content), chunk_id)

    return sorted([ev for ev in items if isinstance(ev, dict)], key=_score, reverse=True)


def _extract_question_scope_terms(question: str) -> List[str]:
    """Best-effort extraction of query scope anchors from the question.

    Language-agnostic heuristic: prefer quoted spans and long unicode word runs.
    """

    q = str(question or "").strip()
    if not q:
        return []

    candidates: list[str] = []
    # Quoted spans (covers a few common quotation styles without relying on a specific language).
    for pat in (r"《([^》]{1,120})》", r"\"([^\"]{1,120})\"", r"“([^”]{1,120})”", r"'([^']{1,120})'", r"‘([^’]{1,120})’"):
        for match in re.finditer(pat, q):
            token = (match.group(1) or "").strip()
            if token and token not in candidates:
                candidates.append(token)

    # Long unicode word runs (letters/digits) optionally with simple separators.
    for match in re.finditer(r"[\w\-\(\)\.]{4,60}", q, flags=re.UNICODE):
        token = match.group(0).strip()
        if not token:
            continue
        # Avoid purely numeric strings.
        if not any(ch.isalpha() for ch in token):
            continue
        if token not in candidates:
            candidates.append(token)

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = normalize_filename_token(raw)
        if not token:
            continue
        # Drop common instruction tokens that frequently appear in prompts but are not part of user scope.
        low_raw = token.lower()
        if "chunk_id" in low_raw or "chunkid" in low_raw:
            continue
        if "not found in evidence" in low_raw:
            continue
        low = token.lower()
        if low in seen:
            continue
        seen.add(low)
        normalized.append(token)
        if len(normalized) >= 20:
            break

    # Prune overly-generic tokens: if a token is contained in a longer token, keep the longer one.
    lowered = [t.lower() for t in normalized]
    kept: list[str] = []
    for idx, token in enumerate(normalized):
        t_low = lowered[idx]
        if any((other != t_low) and (t_low in other) and (len(other) >= len(t_low) + 2) for other in lowered):
            continue
        kept.append(token)
    return kept[:10]


def _filter_evidences_by_question_scope(question: str, items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    terms = _extract_question_scope_terms(question)
    if len(terms) < 2:
        return items, {"scope_terms_applied": False}

    # Prefer terms that are discriminative across the retrieved evidence pool.
    # This keeps cross-doc questions from "drifting" into unrelated files when generic
    # words (e.g. "payment", "plan") appear everywhere.
    evidence_count = len([ev for ev in items if isinstance(ev, dict)])
    max_common_ratio = 0.65
    rare_ratio_cutoff = 0.45
    max_effective_terms = 6
    term_stats: Dict[str, Dict[str, Any]] = {}
    for term in terms:
        term_lower = term.lower()
        hits = 0
        for ev in items:
            if not isinstance(ev, dict):
                continue
            content = str(ev.get("content") or "").lower()
            prov = ev.get("provenance") if isinstance(ev.get("provenance"), dict) else {}
            meta = prov.get("metadata") if isinstance(prov, dict) and isinstance(prov.get("metadata"), dict) else {}
            filename = None
            if isinstance(meta, dict):
                filename = meta.get("filename") or meta.get("source_file_name") or meta.get("path")
                if not filename:
                    chunk_meta = meta.get("chunk_metadata")
                    if isinstance(chunk_meta, dict):
                        filename = chunk_meta.get("filename") or chunk_meta.get("source_file_name") or chunk_meta.get("path")
            filename_lower = str(filename or "").lower()
            if term_lower and (term_lower in content or term_lower in filename_lower):
                hits += 1
        ratio = (hits / evidence_count) if evidence_count else 0.0
        term_stats[term] = {"hits": hits, "ratio": round(ratio, 3)}

    # Guardrail: if the question includes at least one high-specificity anchor (numbers / roman numerals / parentheses)
    # and we cannot find it in the evidence pool at all, do NOT fall back to unrelated evidences.
    def _looks_like_anchor(term: str) -> bool:
        if re.search(r"\d", term):
            return True
        # Roman numerals as standalone markers (avoid matching common letters inside words).
        if re.search(r"(?:^|[^A-Za-z])[IVX]{1,6}(?:$|[^A-Za-z])", term):
            return True
        # Parentheses alone are often prompt scaffolding (e.g. "(with evidence)") and should not
        # hard-filter cross-language evidence pools. Keep parentheses as anchors only when paired
        # with other high-specificity signals (digits / roman numerals already handled above).
        return False

    anchor_terms = [t for t in terms if _looks_like_anchor(t)]
    if anchor_terms:
        if all(int(term_stats.get(t, {}).get("hits") or 0) <= 0 for t in anchor_terms):
            return [], {
                "scope_terms_applied": True,
                "hard_filter": True,
                "anchor_miss": True,
                "anchor_terms": anchor_terms,
                "scope_terms": terms,
                "term_stats": term_stats,
            }

    dropped_missing = [t for t in terms if int(term_stats.get(t, {}).get("hits") or 0) <= 0]
    dropped_common = [t for t in terms if t not in dropped_missing and float(term_stats.get(t, {}).get("ratio") or 0.0) > max_common_ratio]
    candidates = [t for t in terms if t not in dropped_missing and t not in dropped_common]
    if len(candidates) < 2:
        return items, {
            "scope_terms_applied": False,
            "scope_terms": terms,
            "term_stats": term_stats,
            "dropped_missing": dropped_missing,
            "dropped_common": dropped_common,
            "fallback": True,
        }

    # Take the rarest terms first (ratio asc, then prefer longer tokens).
    ranked = sorted(
        candidates,
        key=lambda t: (float(term_stats.get(t, {}).get("ratio") or 1.0), -len(t)),
    )
    rare = [t for t in ranked if float(term_stats.get(t, {}).get("ratio") or 1.0) <= rare_ratio_cutoff]
    if len(rare) >= 2:
        discriminative_terms = rare[:max_effective_terms]
    else:
        discriminative_terms = ranked[: max(2, min(max_effective_terms, len(ranked)))]
    if len(discriminative_terms) < 2:
        return items, {
            "scope_terms_applied": False,
            "scope_terms": terms,
            "term_stats": term_stats,
            "dropped_missing": dropped_missing,
            "dropped_common": dropped_common,
            "fallback": True,
        }

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for ev in items:
        if not isinstance(ev, dict):
            continue
        content = str(ev.get("content") or "").lower()
        prov = ev.get("provenance") if isinstance(ev.get("provenance"), dict) else {}
        meta = prov.get("metadata") if isinstance(prov, dict) and isinstance(prov.get("metadata"), dict) else {}
        filename = None
        if isinstance(meta, dict):
            filename = meta.get("filename") or meta.get("source_file_name") or meta.get("path")
            if not filename:
                chunk_meta = meta.get("chunk_metadata")
                if isinstance(chunk_meta, dict):
                    filename = chunk_meta.get("filename") or chunk_meta.get("source_file_name") or chunk_meta.get("path")
        filename_lower = str(filename or "").lower()
        in_scope = any(term.lower() in content or term.lower() in filename_lower for term in discriminative_terms)
        if in_scope:
            kept.append(ev)
        else:
            dropped += 1

    if kept:
        return kept, {
            "scope_terms_applied": True,
            "scope_terms": terms,
            "scope_terms_effective": discriminative_terms,
            "term_stats": term_stats,
            "dropped_missing": dropped_missing,
            "dropped_common": dropped_common,
            "kept": len(kept),
            "dropped": dropped,
        }
    # Avoid false negatives: if heuristic drops everything, keep original set.
    return items, {
        "scope_terms_applied": False,
        "scope_terms": terms,
        "scope_terms_effective": discriminative_terms,
        "term_stats": term_stats,
        "dropped_missing": dropped_missing,
        "dropped_common": dropped_common,
        "kept": len(items),
        "dropped": 0,
        "fallback": True,
    }


def _is_tool_generated_evidence(evidence: Dict[str, Any], *, toolish_chunk_id_prefixes: tuple[str, ...], toolish_source_names: set[str]) -> bool:
    chunk_id = str(evidence.get("chunk_id") or evidence.get("evidence_id") or "").strip()
    if not chunk_id:
        return False
    lowered = chunk_id.lower()
    if ":" in chunk_id or any(lowered.startswith(prefix) for prefix in toolish_chunk_id_prefixes):
        return True
    source = str(evidence.get("source") or "").strip().lower()
    return source in toolish_source_names


def _split_authoritative_evidences(evidences: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split evidences into authoritative corpus chunks vs tool-generated artifacts."""

    authoritative: List[Dict[str, Any]] = []
    generated: List[Dict[str, Any]] = []
    for ev in evidences or []:
        if not isinstance(ev, dict):
            continue
        if _is_tool_generated_evidence(
            ev,
            toolish_chunk_id_prefixes=composer_defaults.TOOLISH_CHUNK_ID_PREFIXES,
            toolish_source_names=composer_defaults.TOOLISH_SOURCE_NAMES,
        ):
            generated.append(ev)
            continue
        authoritative.append(ev)
    return authoritative, generated


def _trim_text(value: Any, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    limit = max(0, int(max_chars))
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _slim_diagnostics(
    payload: Any,
    *,
    max_keys: int = composer_defaults.DEFAULT_DIAGNOSTICS_MAX_KEYS,
    max_value_chars: int = composer_defaults.DEFAULT_DIAGNOSTICS_MAX_VALUE_CHARS,
) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    trimmed: Dict[str, Any] = {}
    for key in sorted(payload.keys(), key=lambda k: str(k))[: max(0, int(max_keys))]:
        raw = payload.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float, bool)):
            trimmed[str(key)] = raw
        elif isinstance(raw, str):
            trimmed[str(key)] = _trim_text(raw, max_chars=max_value_chars)
        elif isinstance(raw, (list, tuple)) and len(raw) <= composer_defaults.DEFAULT_DIAGNOSTICS_MAX_SMALL_LIST_ITEMS:
            item_limit = composer_defaults.DEFAULT_DIAGNOSTICS_LIST_ITEM_PREVIEW_CHARS
            trimmed[str(key)] = [(_trim_text(item, max_chars=item_limit) if isinstance(item, str) else item) for item in raw]
        else:
            continue
    return trimmed or None


__all__ = [
    "_filter_evidences_by_question_scope",
    "_rank_evidences_for_question",
    "_slim_diagnostics",
    "_split_authoritative_evidences",
    "_trim_text",
]

