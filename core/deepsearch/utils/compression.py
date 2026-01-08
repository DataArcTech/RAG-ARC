"""Shared evidence compaction helpers for DeepSearch.

Goal: one place to implement:
- request-scoped `compression` schema parsing
- tool context window trimming (max_items, max_chars, retention)
- think-mode excerpting (optional) without scattering `[:200]` / `[:1600]` across tools
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext


@dataclass(frozen=True)
class EvidenceCompactionConfig:
    mode: str  # truncate | excerpt
    max_items: int
    max_chars: int
    excerpt_chars: int
    retention: str  # head | tail

    def as_meta(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "max_items": self.max_items,
            "max_chars": self.max_chars,
            "excerpt_chars": (self.excerpt_chars if self.mode == "excerpt" else None),
            "retention": self.retention,
        }


_EXCERPT_TERM_RE = re.compile(r"[A-Za-z]{3,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,}")

_RISK_TERMS: Tuple[str, ...] = (
    "风险",
    "風險",
    "警告",
    "提示",
    "不保证",
    "不保證",
    "并非保证",
    "並非保證",
    "可能",
    "或会",
    "或會",
    "波动",
    "波動",
    "影响",
    "影響",
    "不利",
    "不利影响",
    "not guaranteed",
    "risk",
    "warning",
)


def _coerce_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception:
        parsed = int(default)
    return max(min_value, min(int(parsed), max_value))


def _extract_compression_container(*, graph_context: GraphQueryContext | None, extra: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    if graph_context and isinstance(graph_context.metadata, dict):
        meta = graph_context.metadata
        container = meta.get("compression")
        if isinstance(container, dict):
            merged.update(dict(container))
        req = meta.get("request_metadata")
        if isinstance(req, dict) and isinstance(req.get("compression"), dict):
            # Back-compat: older pipeline stores request metadata under a bucket.
            merged.update(dict(req.get("compression") or {}))

    if isinstance(extra, Mapping):
        container = extra.get("compression")
        if isinstance(container, dict):
            merged.update(dict(container))

    # Accept nested: {"compression": {"think": {...}}} merged above already.
    return merged


def _branch_payload(container: Dict[str, Any], branch: str) -> Dict[str, Any]:
    if not container:
        return {}
    raw = container.get(branch)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def resolve_compaction_config(
    *,
    branch: str,
    graph_context: GraphQueryContext | None,
    extra: Mapping[str, Any] | None,
    default_max_items: int,
    default_max_chars: int,
    default_mode: str = "truncate",
    default_excerpt_chars: int = 900,
    default_retention: str = "tail",
    env_max_items: str | None = None,
    env_max_chars: str | None = None,
    env_excerpt_chars: str | None = None,
) -> EvidenceCompactionConfig:
    """Resolve a branch config from `compression` schema (graph_context + extra) with fallbacks."""

    container = _extract_compression_container(graph_context=graph_context, extra=extra)
    branch_cfg = _branch_payload(container, branch)

    # Back-compat: when the container is a flat object, treat it as `think` overrides only.
    if not branch_cfg and branch == "think":
        branch_cfg = {
            key: value
            for key, value in container.items()
            if key in {"mode", "max_items", "max_chars", "excerpt_chars", "retention"}
        }

    # Defaults can be overridden by env for legacy knobs (think/tool_context).
    max_items_default = default_max_items
    max_chars_default = default_max_chars
    excerpt_default = default_excerpt_chars
    if env_max_items:
        raw = os.getenv(env_max_items, "").strip()
        if raw:
            max_items_default = _coerce_int(raw, default=max_items_default, min_value=0, max_value=64)
    if env_max_chars:
        raw = os.getenv(env_max_chars, "").strip()
        if raw:
            max_chars_default = _coerce_int(raw, default=max_chars_default, min_value=100, max_value=20000)
    if env_excerpt_chars:
        raw = os.getenv(env_excerpt_chars, "").strip()
        if raw:
            excerpt_default = _coerce_int(raw, default=excerpt_default, min_value=200, max_value=6000)

    mode = str(branch_cfg.get("mode") or default_mode).strip().lower() or "truncate"
    if mode not in {"truncate", "excerpt"}:
        mode = "truncate"

    retention = str(branch_cfg.get("retention") or default_retention).strip().lower() or "tail"
    if retention not in {"head", "tail"}:
        retention = "tail"

    max_items = _coerce_int(
        branch_cfg.get("max_items"),
        default=max_items_default,
        min_value=0,
        max_value=64,
    )
    max_chars = _coerce_int(
        branch_cfg.get("max_chars"),
        default=max_chars_default,
        min_value=100,
        max_value=20000,
    )
    excerpt_chars = _coerce_int(
        branch_cfg.get("excerpt_chars"),
        default=excerpt_default,
        min_value=200,
        max_value=6000,
    )
    return EvidenceCompactionConfig(
        mode=mode,
        max_items=max_items,
        max_chars=max_chars,
        excerpt_chars=excerpt_chars,
        retention=retention,
    )


def truncate_text(value: str, *, max_chars: int) -> str:
    text = str(value or "")
    limit = max(0, int(max_chars))
    if not limit or len(text) <= limit:
        return text
    if limit <= 1:
        return "…"
    return text[: max(0, limit - 1)] + "…"


def focused_truncate_text(
    value: str,
    *,
    max_chars: int,
    question: str,
    extra: Mapping[str, Any] | None,
    pre_context_ratio: float = 0.35,
) -> str:
    """Truncate text to a bounded window while trying to preserve query-matching spans.

    This avoids the common failure mode where a long chunk "hits" but the report stage only sees
    the head of the chunk (and thus misses the answer sentence located later).
    """

    raw = str(value or "")
    limit = max(0, int(max_chars))
    if not limit or len(raw) <= limit:
        return raw

    q = str(question or "").strip()
    if not q:
        return truncate_text(raw, max_chars=limit)

    terms = _extract_excerpt_terms(question=q, extra=extra)
    if not terms:
        return truncate_text(raw, max_chars=limit)

    lowered = raw.lower()
    candidates: list[int] = []
    for term in terms:
        needle = str(term or "").strip().lower()
        if not needle:
            continue
        idx = lowered.find(needle)
        if idx >= 0:
            candidates.append(idx)
        if len(candidates) >= 32:
            break

    if not candidates:
        return truncate_text(raw, max_chars=limit)

    ratio = float(pre_context_ratio)
    if ratio < 0.0 or ratio > 1.0:
        ratio = 0.35
    pre_chars = max(0, min(limit, int(limit * ratio)))

    def _window_for(idx: int) -> tuple[int, int]:
        start = max(0, idx - pre_chars)
        end = min(len(raw), start + limit)
        # shift start back if we're near EOF
        start = max(0, end - limit)
        return start, end

    # Pick the best window by unique term coverage (tie: earlier start).
    best_start, best_end = _window_for(candidates[0])
    best_score = -1
    unique_terms = [str(t).strip().lower() for t in terms if str(t).strip()]
    unique_terms = [t for t in unique_terms if t]
    for idx in candidates:
        start, end = _window_for(idx)
        segment_lower = lowered[start:end]
        matched = 0
        seen: set[str] = set()
        for term in unique_terms:
            if term in seen:
                continue
            if term in segment_lower:
                matched += 1
                seen.add(term)
        score = matched
        if score > best_score or (score == best_score and start < best_start):
            best_score = score
            best_start, best_end = start, end

    prefix = best_start > 0
    suffix = best_end < len(raw)
    # Keep the final string within `limit` after adding ellipses.
    allowance = limit - (1 if prefix else 0) - (1 if suffix else 0)
    allowance = max(0, allowance)
    body = raw[best_start:best_end]
    if allowance and len(body) > allowance:
        body = body[:allowance]

    out = body
    if prefix:
        out = "…" + out
    if suffix:
        out = out + "…"
    if len(out) > limit:
        out = truncate_text(out, max_chars=limit)
    return out


def _extract_excerpt_terms(*, question: str, extra: Mapping[str, Any] | None) -> List[str]:
    q = str(question or "")
    terms: List[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        norm = (token or "").strip()
        if not norm:
            return
        key = norm.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(norm)

    for match in _EXCERPT_TERM_RE.finditer(q):
        _add(match.group(0))
        if len(terms) >= 24:
            break

    triples = None
    if isinstance(extra, Mapping):
        triples = extra.get("triples")
    if isinstance(triples, list):
        for item in triples:
            if not isinstance(item, dict):
                continue
            for key in ("subject", "predicate", "object", "s", "p", "o", "head", "relation", "tail"):
                if isinstance(item.get(key), str):
                    _add(item.get(key, ""))  # type: ignore[arg-type]
                    if len(terms) >= 32:
                        return terms
    return terms


def _first_index(lowered: str, needles: Iterable[str]) -> Optional[int]:
    best = None
    for term in needles:
        needle = str(term or "").strip().lower()
        if not needle:
            continue
        idx = lowered.find(needle)
        if idx < 0:
            continue
        if best is None or idx < best:
            best = idx
            if best == 0:
                break
    return best


def _excerpt_content(text: str, *, terms: List[str], excerpt_chars: int) -> str:
    raw = str(text or "")
    if not raw or not terms or excerpt_chars <= 0:
        return raw
    lowered = raw.lower()
    idxs: List[int] = []

    primary = _first_index(lowered, terms)
    if primary is not None:
        idxs.append(primary)
    risk = _first_index(lowered, _RISK_TERMS)
    if risk is not None and (primary is None or abs(risk - primary) > 120):
        idxs.append(risk)

    if not idxs:
        return raw

    windows = sorted(set(idxs))[:2]
    per_window = max(200, int(excerpt_chars / max(1, len(windows))))
    parts: List[str] = []
    for idx in windows:
        half = max(1, int(per_window / 2))
        start = max(0, idx - half)
        end = min(len(raw), idx + half)
        part = raw[start:end].strip()
        if start > 0:
            part = "…" + part
        if end < len(raw):
            part = part + "…"
        parts.append(part)
    return "\n".join(parts)


def compact_evidences(
    evidences: Sequence[EvidenceChunk],
    *,
    cfg: EvidenceCompactionConfig,
    question: str,
    extra: Mapping[str, Any] | None,
    include_triple_count: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return compacted evidence dicts (chunk_id/source/content/score + optional triple_count)."""

    items = list(evidences or [])
    if cfg.max_items and len(items) > cfg.max_items:
        if cfg.retention == "head":
            items = items[: cfg.max_items]
        else:
            items = items[-cfg.max_items :]

    terms = _extract_excerpt_terms(question=question, extra=extra) if cfg.mode == "excerpt" else []
    out: List[Dict[str, Any]] = []
    for ev in items:
        content = str(getattr(ev, "content", "") or "").strip()
        if cfg.mode == "excerpt":
            content = _excerpt_content(content, terms=terms, excerpt_chars=cfg.excerpt_chars)
        content = truncate_text(content, max_chars=cfg.max_chars)
        entry: Dict[str, Any] = {
            "chunk_id": getattr(ev, "chunk_id", None),
            "source": getattr(ev, "source", None),
            "content": content,
            "score": getattr(ev, "score", None),
        }
        if include_triple_count:
            prov = getattr(ev, "provenance", None)
            if isinstance(prov, dict):
                triples = prov.get("triples")
                if isinstance(triples, list) and triples:
                    entry["triple_count"] = len(triples)
        out.append(entry)

    meta = cfg.as_meta()
    if cfg.mode == "excerpt":
        meta["excerpt_terms"] = terms[:8]
    return out, meta


def compact_context_snippet(
    evidences: Sequence[EvidenceChunk],
    *,
    cfg: EvidenceCompactionConfig,
    question: str,
    extra: Mapping[str, Any] | None,
    joiner: str = "\n",
) -> Tuple[str, Dict[str, Any]]:
    compacted, meta = compact_evidences(evidences, cfg=cfg, question=question, extra=extra, include_triple_count=False)
    snippet = joiner.join(str(item.get("content") or "") for item in compacted if str(item.get("content") or "").strip())
    return snippet, meta


def safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)
