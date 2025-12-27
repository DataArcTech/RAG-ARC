"""Consistency checking for DeepSearch reports.

This module provides a lightweight, opt-in quality gate that can be used to
detect obvious citation/evidence mismatches and (optionally) ask an LLM to
judge whether claims are supported by the provided evidence.
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from core.deepsearch.tools.base import call_llm_async
from core.prompts.deepsearch.report import (
    CONSISTENCY_CHECK_SYSTEM_PROMPT,
    CONSISTENCY_CHECK_USER_PROMPT,
)

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_CJK_BRACKET_RE = re.compile(r"【([^【】]+)】")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?\u3002\uff01\uff1f])\s+")


class ConsistencyIssue(BaseModel):
    issue_type: str = Field(..., description="unsupported_claim|misquote|contradiction|unknown_citation|checker_error")
    location: str = Field(..., description="Where the issue appears (e.g. 'summary', 'section:2', 'evidence_index').")
    description: str = Field(..., min_length=1)
    suggested_fix: Optional[str] = None


class ConsistencyCheckResult(BaseModel):
    is_consistent: bool
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ConsistencyChecker:
    """Check the report for consistency against provided evidence."""

    def __init__(self, llm_connector: Any, *, temperature: float = 0.0, max_retries: int = 2) -> None:
        self.llm_connector = llm_connector
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)

    async def check(
        self,
        *,
        question: str,
        report_markdown: str,
        evidences: Sequence[Dict[str, Any]],
        structured_report: Optional[Dict[str, Any]] = None,
        max_evidence_chars: int = 900,
        max_claims: int = 40,
    ) -> ConsistencyCheckResult:
        local_result = self._local_checks(report_markdown, evidences, structured_report)
        llm_result = await self._llm_judge(
            question=question,
            report_markdown=report_markdown,
            evidences=evidences,
            max_evidence_chars=max_evidence_chars,
            structured_report=structured_report,
            max_claims=max_claims,
        )
        return self._merge_results(local_result, llm_result)

    @staticmethod
    def _merge_results(
        local_result: ConsistencyCheckResult, llm_result: Optional[ConsistencyCheckResult]
    ) -> ConsistencyCheckResult:
        if llm_result is None:
            return local_result
        issues = list(local_result.issues) + list(llm_result.issues)
        is_consistent = local_result.is_consistent and llm_result.is_consistent
        confidence = min(local_result.confidence, llm_result.confidence)
        return ConsistencyCheckResult(is_consistent=is_consistent, issues=issues, confidence=confidence)

    @staticmethod
    def _local_checks(
        report_markdown: str,
        evidences: Sequence[Dict[str, Any]],
        structured_report: Optional[Dict[str, Any]],
    ) -> ConsistencyCheckResult:
        evidence_ids = {str(ev.get("chunk_id") or "") for ev in evidences if isinstance(ev, dict)}
        evidence_ids.discard("")

        issues: List[ConsistencyIssue] = []
        citations = (structured_report or {}).get("citations") if isinstance(structured_report, dict) else None
        if isinstance(citations, list):
            for idx, entry in enumerate(citations):
                if not isinstance(entry, dict):
                    continue
                ev_id = str(entry.get("evidence_id") or "").strip()
                if ev_id and ev_id not in evidence_ids:
                    issues.append(
                        ConsistencyIssue(
                            issue_type="unknown_citation",
                            location=f"evidence_index:{idx + 1}",
                            description=f"Citation references unknown evidence_id='{ev_id}'.",
                            suggested_fix="Remove the citation or add the corresponding evidence item.",
                        )
                    )

        # If we find unknown citations we treat it as inconsistent; otherwise stay neutral.
        is_consistent = not issues
        confidence = 0.9 if issues else 1.0
        return ConsistencyCheckResult(is_consistent=is_consistent, issues=issues, confidence=confidence)

    async def _llm_judge(
        self,
        *,
        question: str,
        report_markdown: str,
        evidences: Sequence[Dict[str, Any]],
        max_evidence_chars: int,
        structured_report: Optional[Dict[str, Any]],
        max_claims: int,
    ) -> Optional[ConsistencyCheckResult]:
        if self.llm_connector is None:
            raise RuntimeError("ConsistencyChecker requires an LLM connector")

        claims = _extract_cited_claims(
            structured_report=structured_report or {},
            evidences=list(evidences),
            max_evidence_chars=max_evidence_chars,
            max_claims=max_claims,
        )
        user_prompt = CONSISTENCY_CHECK_USER_PROMPT.format(question=question, claims_json=json.dumps(claims, ensure_ascii=False, indent=2))
        messages = [
            {"role": "system", "content": CONSISTENCY_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_exc: Exception | None = None
        for attempt in range(max(self.max_retries, 1)):
            try:
                raw = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await asyncio.sleep(min(8.0, 2.0**attempt))
                continue

            parsed = _safe_parse_json(raw)
            if parsed is None:
                raise ValueError("Consistency judge did not return valid JSON.")
            try:
                return ConsistencyCheckResult.model_validate(parsed)
            except ValidationError:
                raise ValueError("Consistency judge returned JSON with an unexpected schema.") from None
        raise RuntimeError(f"Consistency judge failed to run: {last_exc}") from last_exc


def _safe_parse_json(raw: str) -> Dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    extracted = _extract_first_json(text)
    if extracted is None:
        return None
    try:
        value = json.loads(extracted)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_first_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_cited_claims(
    *,
    structured_report: Dict[str, Any],
    evidences: List[Dict[str, Any]],
    max_evidence_chars: int,
    max_claims: int,
) -> List[Dict[str, Any]]:
    evidence_lookup: Dict[str, Dict[str, Any]] = {}
    for entry in evidences:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        if chunk_id:
            evidence_lookup[chunk_id] = entry
    known_ids = set(evidence_lookup)

    def _iter_text_blocks() -> List[tuple[str, str]]:
        blocks: List[tuple[str, str]] = []
        summary = structured_report.get("summary")
        if isinstance(summary, str) and summary.strip():
            blocks.append(("summary", summary.strip()))
        sections = structured_report.get("sections")
        if isinstance(sections, list):
            for idx, section in enumerate(sections, start=1):
                if not isinstance(section, dict):
                    continue
                title = str(section.get("title") or "").strip()
                body = str(section.get("body_markdown") or "").strip()
                if not body:
                    continue
                label = f"section:{idx}" if not title else f"section:{idx}:{title}"
                blocks.append((label, body))
        return blocks

    def _iter_sentences(text: str) -> List[str]:
        normalized = (text or "").strip()
        if not normalized:
            return []
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        sentences: List[str] = []
        for line in lines:
            line = re.sub(r"^#{1,6}\\s+", "", line).strip()
            line = re.sub(r"^[-*]\\s+", "", line).strip()
            if not line:
                continue
            parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(line) if part.strip()]
            sentences.extend(parts if parts else [line])
        return sentences

    def _extract_citations(sentence: str) -> List[str]:
        raw_tokens: List[str] = []
        raw_tokens.extend(_BRACKET_RE.findall(sentence))
        raw_tokens.extend(_CJK_BRACKET_RE.findall(sentence))
        ids: List[str] = []
        seen: set[str] = set()
        for raw in raw_tokens:
            for candidate in re.split(r"[,\s]+", raw.strip()):
                token = candidate.strip()
                if not token or token in seen:
                    continue
                if token in known_ids:
                    seen.add(token)
                    ids.append(token)
        return ids

    out: List[Dict[str, Any]] = []
    for location, text in _iter_text_blocks():
        for sentence in _iter_sentences(text):
            cited_ids = _extract_citations(sentence)
            if not cited_ids:
                continue
            snippets: List[Dict[str, Any]] = []
            for ev_id in cited_ids:
                ev = evidence_lookup.get(ev_id) or {}
                content = str(ev.get("content") or "")
                if max_evidence_chars > 0 and len(content) > max_evidence_chars:
                    content = content[:max_evidence_chars].rstrip() + "…"
                snippets.append({"chunk_id": ev_id, "source": ev.get("source"), "content": content})
            out.append(
                {
                    "location": location,
                    "claim": sentence.strip(),
                    "citations": cited_ids,
                    "evidence_snippets": snippets,
                }
            )
            if max_claims > 0 and len(out) >= max_claims:
                return out
    return out
