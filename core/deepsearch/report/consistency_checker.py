"""Consistency checking for DeepSearch reports.

This module provides a lightweight, opt-in quality gate that can be used to
detect obvious citation/evidence mismatches and (optionally) ask an LLM to
judge whether claims are supported by the provided evidence.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from core.deepsearch.tools.base import call_llm_async
from core.prompts.deepsearch.report import (
    CONSISTENCY_CHECK_SYSTEM_PROMPT,
    CONSISTENCY_CHECK_USER_PROMPT,
)


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
        report_markdown: str,
        evidences: Sequence[Dict[str, Any]],
        structured_report: Optional[Dict[str, Any]] = None,
        max_evidence_items: int | None = 24,
        max_evidence_chars: int = 900,
    ) -> ConsistencyCheckResult:
        local_result = self._local_checks(report_markdown, evidences, structured_report)
        llm_result = await self._llm_judge(
            report_markdown=report_markdown,
            evidences=evidences,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
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
        report_markdown: str,
        evidences: Sequence[Dict[str, Any]],
        max_evidence_items: int | None,
        max_evidence_chars: int,
    ) -> Optional[ConsistencyCheckResult]:
        if self.llm_connector is None:
            return None

        limited_evidences = _limit_evidences(list(evidences), max_evidence_items, max_evidence_chars)
        user_prompt = CONSISTENCY_CHECK_USER_PROMPT.format(
            report_markdown=report_markdown,
            evidence_json=json.dumps(limited_evidences, ensure_ascii=False, indent=2),
        )
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
                return ConsistencyCheckResult(
                    is_consistent=False,
                    issues=[
                        ConsistencyIssue(
                            issue_type="checker_error",
                            location="checker",
                            description="Consistency judge did not return valid JSON.",
                            suggested_fix="Retry consistency checking or disable it for this run.",
                        )
                    ],
                    confidence=0.0,
                )
            try:
                return ConsistencyCheckResult.model_validate(parsed)
            except ValidationError:
                return ConsistencyCheckResult(
                    is_consistent=False,
                    issues=[
                        ConsistencyIssue(
                            issue_type="checker_error",
                            location="checker",
                            description="Consistency judge returned JSON with an unexpected schema.",
                            suggested_fix="Retry consistency checking or disable it for this run.",
                        )
                    ],
                    confidence=0.0,
                )
        if last_exc is not None:
            return ConsistencyCheckResult(
                is_consistent=False,
                issues=[
                    ConsistencyIssue(
                        issue_type="checker_error",
                        location="checker",
                        description=f"Consistency judge failed to run: {last_exc}",
                        suggested_fix="Retry consistency checking or disable it for this run.",
                    )
                ],
                confidence=0.0,
            )
        return None


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


def _limit_evidences(evidences: List[Dict[str, Any]], max_items: int | None, max_chars: int) -> List[Dict[str, Any]]:
    subset = list(evidences) if max_items is None else evidences[: max(max_items, 0)]
    limited: List[Dict[str, Any]] = []
    for entry in subset:
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content") or "")
        if max_chars > 0 and len(content) > max_chars:
            content = content[: max_chars].rstrip() + "…"
        cloned = dict(entry)
        cloned["content"] = content
        limited.append(cloned)
    return limited

