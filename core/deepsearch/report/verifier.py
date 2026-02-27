"""Post-report numerical accuracy verifier for DeepSearch.

Calls the LLM to verify arithmetic, value tracing, and internal consistency
of a generated report before returning it to the user. If verification fails,
the caller can retry report generation with the verification feedback.
"""

import logging
from typing import Any, Dict, List, Optional

from core.prompts.deepsearch.report import (
    REPORT_VERIFY_SYSTEM_PROMPT_EN,
    REPORT_VERIFY_USER_PROMPT_EN,
)
from core.utils.llm_json import call_llm_json_with_retry

logger = logging.getLogger(__name__)


class ReportVerificationResult:
    """Outcome of the report verification step."""

    __slots__ = ("passed", "issues", "corrected_answer", "thinking", "raw")

    def __init__(
        self,
        *,
        passed: bool,
        issues: List[Dict[str, str]],
        corrected_answer: Optional[str] = None,
        thinking: str = "",
        raw: Any = None,
    ):
        self.passed = passed
        self.issues = issues
        self.corrected_answer = corrected_answer
        self.thinking = thinking
        self.raw = raw

    def feedback_text(self) -> str:
        """Build a concise feedback string suitable for injecting into a report-rewrite prompt."""
        if self.passed:
            return ""
        lines = ["## Verification Feedback (MUST FIX before answering)"]
        for issue in self.issues:
            itype = issue.get("type", "error")
            desc = issue.get("description", "")
            lines.append(f"- [{itype}] {desc}")
        if self.corrected_answer:
            lines.append(f"- Corrected answer: {self.corrected_answer}")
        return "\n".join(lines)


async def verify_report(
    *,
    llm_connector: Any,
    question: str,
    report_text: str,
    evidence_pack: str,
    temperature: float = 0.0,
) -> ReportVerificationResult:
    """Run a single-shot verification LLM call.

    Returns a :class:`ReportVerificationResult` regardless of whether the LLM
    succeeds or fails (on LLM/parse error, ``passed`` defaults to ``True`` so
    that the pipeline is not blocked by verifier failures).
    """

    if not report_text or not report_text.strip():
        return ReportVerificationResult(passed=True, issues=[], thinking="empty report")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": REPORT_VERIFY_SYSTEM_PROMPT_EN},
        {
            "role": "user",
            "content": REPORT_VERIFY_USER_PROMPT_EN.format(
                question=question,
                report_text=report_text,
                evidence_pack=evidence_pack,
            ),
        },
    ]

    try:
        result = await call_llm_json_with_retry(
            llm_connector=llm_connector,
            messages=messages,
            expected="dict",
            temperature=temperature,
            attempts=1,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Report verifier LLM call failed; assuming pass", exc_info=True)
        return ReportVerificationResult(passed=True, issues=[], thinking="verifier_error")

    if not isinstance(result, dict):
        logger.warning("Report verifier returned non-dict: %s", type(result))
        return ReportVerificationResult(passed=True, issues=[], thinking="parse_error")

    passed = bool(result.get("passed", True))
    issues = result.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    corrected = result.get("corrected_answer")
    thinking = str(result.get("thinking") or "")

    return ReportVerificationResult(
        passed=passed,
        issues=issues,
        corrected_answer=str(corrected) if corrected else None,
        thinking=thinking,
        raw=result,
    )
