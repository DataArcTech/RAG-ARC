"""Benchmark-mode answer synthesis for DeepSearch.

Benchmark mode is used by offline evaluations. This module intentionally avoids product-layer behaviors:
- No long report generation.
- No citation formatting or quality-gate enforcement.
- No refusal/guardrail messaging.

It still consumes the same DeepSearch plan/reasoning traces, so the planning and tool-calling
process remains unchanged; only the final user-facing answer is simplified.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config.core.deepsearch import bench_answer_defaults
from core.deepsearch.utils.compression import focused_truncate_text
from core.deepsearch.utils.evidence_kinds import EvidenceKind, coerce_evidence_kind, count_evidences_by_kind, get_evidence_kind
from core.prompts.deepsearch import (
    DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_COVERAGE_EN,
    DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_EN,
    DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_STRICT_EN,
    DEEPSEARCH_BENCH_ANSWER_USER_PROMPT_TEMPLATE_EN,
    DEEPSEARCH_BENCH_EXTRACT_SYSTEM_PROMPT_EN,
    DEEPSEARCH_BENCH_EXTRACT_USER_PROMPT_TEMPLATE_EN,
    DEEPSEARCH_BENCH_FINAL_SYSTEM_PROMPT_EN,
    DEEPSEARCH_BENCH_FINAL_USER_PROMPT_TEMPLATE_EN,
)
from core.utils.json_extract import safe_json_loads


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _extract_evidence_text(evidence: Any) -> str | None:
    if evidence is None:
        return None
    if isinstance(evidence, str):
        text = evidence.strip()
        return text or None
    if isinstance(evidence, Mapping):
        for key in ("content", "text", "snippet", "prompt_text", "index_text"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    # Pydantic models / objects
    for attr in ("content", "text", "snippet"):
        value = getattr(evidence, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class _BenchEvidenceItem:
    chunk_id: str
    source: str
    content: str
    kind: EvidenceKind
    score: float | None = None


def _coerce_item(evidence: Any) -> _BenchEvidenceItem | None:
    text = _extract_evidence_text(evidence)
    if not text:
        return None

    mapping = _as_mapping(evidence)
    kind = get_evidence_kind(evidence)
    chunk_id = ""
    source = ""
    score: float | None = None
    if mapping:
        chunk_id = str(mapping.get("chunk_id") or mapping.get("id") or "").strip()
        source = str(mapping.get("source") or "").strip()
        kind = coerce_evidence_kind(mapping.get("kind"), default=kind)
        raw_score = mapping.get("score")
        try:
            score = float(raw_score) if raw_score is not None else None
        except Exception:
            score = None
    else:
        chunk_id = str(getattr(evidence, "chunk_id", "") or "").strip()
        source = str(getattr(evidence, "source", "") or "").strip()
        kind = coerce_evidence_kind(getattr(evidence, "kind", None), default=kind)
        raw_score = getattr(evidence, "score", None)
        try:
            score = float(raw_score) if raw_score is not None else None
        except Exception:
            score = None

    return _BenchEvidenceItem(
        chunk_id=chunk_id,
        source=source,
        content=str(text),
        kind=kind,
        score=score,
    )


def _extract_anchor_terms(question: str) -> List[str]:
    q = str(question or "").strip()
    if not q:
        return []
    anchors: list[str] = []

    for token in re.findall(r"\d[\d,\.%]*", q, flags=re.UNICODE):
        token = token.strip()
        if token and token not in anchors:
            anchors.append(token)
    for token in re.findall(r"[A-Za-z]{3,32}", q, flags=re.UNICODE):
        token = token.strip()
        if token and token not in anchors:
            anchors.append(token)
    for token in re.findall(r"[\u4e00-\u9fff]{2,8}", q, flags=re.UNICODE):
        token = token.strip()
        if token and token not in anchors:
            anchors.append(token)

    return anchors[:24]


def _looks_like_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    if s.endswith(":") and len(s) <= 120:
        return True
    if re.match(r"^[-*]?\s*\*\*[^*]{2,120}\*\*:?\s*$", s):
        return True
    return False


def _looks_like_bullet(line: str) -> bool:
    s = (line or "").lstrip()
    return bool(re.match(r"^(?:[-*]|\d+[.)])\s+\S+", s))


def _strip_leading_bullet_prefix(text: str) -> str:
    """Avoid '- - ...' when we bulletize evidence items."""
    if not text:
        return text
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].lstrip()
    for prefix in ("- ", "* "):
        if first.startswith(prefix):
            lines[0] = first[len(prefix) :]
            break
    return "\n".join(lines).strip()


def _snippet_for_evidence(
    *,
    question: str,
    evidence_text: str,
    snippet_chars: int,
    heading_window_max_lines: int,
) -> str:
    raw = str(evidence_text or "").strip()
    if not raw:
        return ""

    anchors = _extract_anchor_terms(question)
    lines = raw.splitlines()

    best_idx = None
    best_score = 0
    for idx, line in enumerate(lines):
        lower = line.lower()
        hits = sum(1 for a in anchors if a and a.lower() in lower)
        if hits <= 0:
            continue
        if hits > best_score:
            best_score = hits
            best_idx = idx

    if best_idx is not None:
        line = lines[best_idx]
        if _looks_like_heading(line) and any(_looks_like_bullet(n) for n in lines[best_idx + 1 : best_idx + 3]):
            picked: list[str] = []
            for nxt in lines[best_idx : best_idx + max(1, int(heading_window_max_lines))]:
                if not nxt.strip():
                    break
                picked.append(nxt.rstrip())
                if sum(len(p) + 1 for p in picked) >= max(50, int(snippet_chars)):
                    break
            return "\n".join(picked)[: max(0, int(snippet_chars))].strip()

    return focused_truncate_text(raw, max_chars=max(50, int(snippet_chars)), question=question, extra={}).strip()


def _format_evidence_block(
    evidences: Iterable[_BenchEvidenceItem],
    *,
    max_items: int | None,
    max_chars: int | None,
) -> str:
    lines: List[str] = []
    remaining_chars = max_chars if (max_chars is None or max_chars >= 0) else 0
    used = 0
    for idx, evidence in enumerate(evidences):
        if max_items is not None and idx >= max_items:
            break
        text = _strip_leading_bullet_prefix(str(evidence.content or ""))
        if not text.strip():
            continue

        label = f"- {text.strip()}"
        if max_chars is not None:
            remaining = max(0, int(remaining_chars) - used)
            if remaining <= 0:
                break
            if len(label) > remaining:
                label = label[:remaining].rstrip()
        lines.append(label)
        used += len(label) + 1
        if max_chars is not None and used >= int(remaining_chars):
            break

    return "\n".join(lines).strip()


def _resolve_bench_answer_config(
    *,
    bench_config: Mapping[str, Any] | None,
    question_type: str | None,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return (bench_root_cfg, per_type_policy_cfg) dicts with defaults applied."""
    root: Dict[str, Any] = {}
    if isinstance(bench_config, Mapping):
        root.update(dict(bench_config))

    policies = root.get("policies_by_question_type")
    policy_map: Dict[str, Any] = dict(policies) if isinstance(policies, Mapping) else {}
    default_policy = root.get("default_policy")
    default_policy_payload = dict(default_policy) if isinstance(default_policy, Mapping) else {}

    qtype = str(question_type or "").strip()
    if qtype and qtype in policy_map and isinstance(policy_map.get(qtype), Mapping):
        return root, dict(policy_map[qtype])
    return root, default_policy_payload


def _system_prompt_for_preference(preference: str | None) -> str:
    pref = str(preference or "").strip().lower()
    if pref == "correctness":
        return DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_STRICT_EN
    if pref == "coverage":
        return DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_COVERAGE_EN
    return DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_EN


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


async def synthesize_benchmark_answer(
    *,
    llm_connector: Any,
    question: str,
    reasoning_trace: Dict[str, Any],
    external_evidence: Optional[Iterable[Any]] = None,
    question_type: str | None = None,
    bench_answer_config: Mapping[str, Any] | None = None,
    max_evidence_items: int | None = None,  # legacy fallback
    max_evidence_chars: int | None = None,  # legacy fallback
) -> Dict[str, Any]:
    if llm_connector is None:
        raise ValueError("llm_connector is required for benchmark answer synthesis")

    raw_evidences: List[Any] = []
    trace = reasoning_trace or {}
    raw = trace.get("evidences")
    if isinstance(raw, list):
        raw_evidences.extend(raw)
    if external_evidence:
        raw_evidences.extend(list(external_evidence))

    bench_root, policy = _resolve_bench_answer_config(bench_config=bench_answer_config, question_type=question_type)

    allowed_kinds_raw = bench_root.get("allowed_evidence_kinds") or bench_answer_defaults.DEFAULT_BENCH_ALLOWED_EVIDENCE_KINDS
    allowed_kinds = tuple(
        str(k).strip().lower()
        for k in (allowed_kinds_raw if isinstance(allowed_kinds_raw, Sequence) else [])
        if str(k).strip()
    )
    if not allowed_kinds:
        allowed_kinds = tuple(bench_answer_defaults.DEFAULT_BENCH_ALLOWED_EVIDENCE_KINDS)

    max_items_int = _coerce_int_or_none(policy.get("max_evidence_items"))
    max_chars_int = _coerce_int_or_none(policy.get("max_evidence_chars"))
    snippet_chars = _coerce_int_or_none(policy.get("snippet_chars")) or bench_answer_defaults.DEFAULT_BENCH_SNIPPET_CHARS
    heading_window = _coerce_int_or_none(bench_root.get("heading_window_max_lines")) or bench_answer_defaults.DEFAULT_BENCH_HEADING_WINDOW_MAX_LINES

    # Legacy fallback: if a policy didn't specify budgets, preserve old behavior for callers/tests.
    if max_items_int is None and max_evidence_items is not None:
        max_items_int = int(max_evidence_items)
    if max_chars_int is None and max_evidence_chars is not None:
        max_chars_int = int(max_evidence_chars)

    items: List[_BenchEvidenceItem] = []
    for ev in raw_evidences:
        item = _coerce_item(ev)
        if item is None:
            continue
        if item.kind not in allowed_kinds:  # type: ignore[comparison-overlap]
            continue
        items.append(item)

    anchors = _extract_anchor_terms(question)

    def _rank_key(item: _BenchEvidenceItem) -> tuple[int, int, float, int]:
        lowered = item.content.lower()
        hits = sum(1 for a in anchors if a and a.lower() in lowered)
        score = item.score if item.score is not None else 0.0
        return (1 if hits > 0 else 0, hits, score, -len(item.content))

    ranked = sorted(items, key=_rank_key, reverse=True)

    snippet_items: List[_BenchEvidenceItem] = []
    seen_snippets: set[str] = set()
    for item in ranked:
        snippet = _snippet_for_evidence(
            question=question,
            evidence_text=item.content,
            snippet_chars=int(snippet_chars),
            heading_window_max_lines=int(heading_window),
        )
        normalized = snippet.strip()
        if not normalized:
            continue
        if normalized in seen_snippets:
            continue
        seen_snippets.add(normalized)
        snippet_items.append(
            _BenchEvidenceItem(
                chunk_id=item.chunk_id,
                source=item.source,
                content=normalized,
                kind=item.kind,
                score=item.score,
            )
        )

    evidence_block = _format_evidence_block(
        snippet_items,
        max_items=max_items_int,
        max_chars=max_chars_int,
    )

    achat = getattr(llm_connector, "achat", None)
    if not callable(achat):
        raise ValueError("llm_connector does not support async chat (missing .achat)")

    diagnostics: Dict[str, Any] = {
        "question_type": question_type,
        "bench_policy": dict(policy),
        "allowed_evidence_kinds": list(allowed_kinds),
        "raw_evidence_counts_by_kind": count_evidences_by_kind(raw_evidences),
        "evidence_used_count": len(snippet_items),
    }

    mode = str(policy.get("mode") or "single_stage").strip().lower()
    if mode == "two_stage":
        extract_messages = [
            {"role": "system", "content": DEEPSEARCH_BENCH_EXTRACT_SYSTEM_PROMPT_EN},
            {
                "role": "user",
                "content": DEEPSEARCH_BENCH_EXTRACT_USER_PROMPT_TEMPLATE_EN.format(
                    question=(question or "").strip(),
                    evidence=evidence_block or "(no evidence provided)",
                ),
            },
        ]
        extracted_raw = await achat(extract_messages)
        extracted_text = str(extracted_raw or "").strip()
        extracted_payload = safe_json_loads(extracted_text, expected="dict")
        points = extracted_payload.get("points") if isinstance(extracted_payload, dict) else None
        if isinstance(points, list):
            final_messages = [
                {"role": "system", "content": DEEPSEARCH_BENCH_FINAL_SYSTEM_PROMPT_EN},
                {
                    "role": "user",
                    "content": DEEPSEARCH_BENCH_FINAL_USER_PROMPT_TEMPLATE_EN.format(
                        question=(question or "").strip(),
                        points_json=json.dumps({"points": points}, ensure_ascii=False),
                    ),
                },
            ]
            final_answer = await achat(final_messages)
            return {
                "answer": str(final_answer or "").strip(),
                "bench_evidence_block": evidence_block,
                "bench_evidence_used": [item.__dict__ for item in snippet_items],
                "bench_diagnostics": diagnostics,
                "bench_extracted_points": {"points": points},
            }

        # Observable fallback (keep error details for debugging).
        diagnostics["two_stage_error"] = "extract_parse_failed"
        diagnostics["extract_raw_preview"] = extracted_text[:800]

    system_prompt = _system_prompt_for_preference(policy.get("preference"))
    user_prompt = DEEPSEARCH_BENCH_ANSWER_USER_PROMPT_TEMPLATE_EN.format(
        question=(question or "").strip(),
        evidence=evidence_block or "(no evidence provided)",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    answer = await achat(messages)
    answer_text = str(answer or "").strip()
    return {
        "answer": answer_text,
        "bench_evidence_block": evidence_block,
        "bench_evidence_used": [item.__dict__ for item in snippet_items],
        "bench_diagnostics": diagnostics,
    }

