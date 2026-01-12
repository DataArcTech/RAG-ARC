"""Benchmark-mode answer synthesis for DeepSearch.

This intentionally avoids product-layer behaviors:
- No long report generation.
- No citation formatting or quality-gate enforcement.
- No refusal/guardrail messaging.

It still consumes the same DeepSearch plan/reasoning traces, so the planning and tool-calling
process remains unchanged; only the final user-facing answer is simplified.
"""
from typing import Any, Dict, Iterable, List, Optional

from core.prompts.deepsearch import (
    DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_EN,
    DEEPSEARCH_BENCH_ANSWER_USER_PROMPT_TEMPLATE_EN,
)


def _extract_evidence_text(evidence: Any) -> str | None:
    if evidence is None:
        return None
    if isinstance(evidence, str):
        text = evidence.strip()
        return text or None
    if isinstance(evidence, dict):
        for key in ("content", "text", "snippet", "prompt_text", "index_text"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _format_evidence_block(
    evidences: Iterable[Any],
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
        text = _extract_evidence_text(evidence)
        if not text:
            continue

        label = f"- {text}"
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


async def synthesize_benchmark_answer(
    *,
    llm_connector: Any,
    question: str,
    reasoning_trace: Dict[str, Any],
    external_evidence: Optional[Iterable[Any]] = None,
    max_evidence_items: int | None = None,
    max_evidence_chars: int | None = None,
) -> Dict[str, Any]:
    if llm_connector is None:
        raise ValueError("llm_connector is required for benchmark answer synthesis")

    evidences: List[Any] = []
    trace = reasoning_trace or {}
    raw = trace.get("evidences")
    if isinstance(raw, list):
        evidences.extend(raw)
    if external_evidence:
        evidences.extend(list(external_evidence))

    evidence_block = _format_evidence_block(
        evidences,
        max_items=max_evidence_items,
        max_chars=max_evidence_chars,
    )
    user_prompt = DEEPSEARCH_BENCH_ANSWER_USER_PROMPT_TEMPLATE_EN.format(
        question=(question or "").strip(),
        evidence=evidence_block or "(no evidence provided)",
    )
    messages = [
        {"role": "system", "content": DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_EN},
        {"role": "user", "content": user_prompt},
    ]

    achat = getattr(llm_connector, "achat", None)
    if not callable(achat):
        raise ValueError("llm_connector does not support async chat (missing .achat)")
    answer = await achat(messages)
    answer_text = str(answer or "").strip()
    return {"answer": answer_text}

