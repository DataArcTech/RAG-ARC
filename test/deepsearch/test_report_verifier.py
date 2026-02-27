"""Tests for core.deepsearch.report.verifier."""

import pytest

from core.deepsearch.report.verifier import ReportVerificationResult, verify_report


# ── Unit tests for ReportVerificationResult ──────────────────────────────────


class TestReportVerificationResult:
    def test_passed_result_has_empty_feedback(self):
        r = ReportVerificationResult(passed=True, issues=[])
        assert r.passed is True
        assert r.feedback_text() == ""

    def test_failed_result_builds_feedback(self):
        r = ReportVerificationResult(
            passed=False,
            issues=[
                {"type": "arithmetic_error", "description": "6708+1781=8489, not 8275"},
                {"type": "value_mismatch", "description": "equity should be 2795, not 3152"},
            ],
            corrected_answer="8489",
        )
        assert r.passed is False
        fb = r.feedback_text()
        assert "Verification Feedback" in fb
        assert "arithmetic_error" in fb
        assert "6708+1781=8489" in fb
        assert "Corrected answer: 8489" in fb

    def test_failed_without_corrected_answer(self):
        r = ReportVerificationResult(
            passed=False,
            issues=[{"type": "missing_component", "description": "missed lease liabilities"}],
        )
        fb = r.feedback_text()
        assert "missing_component" in fb
        assert "Corrected answer" not in fb


# ── Integration tests for verify_report (mocked LLM) ────────────────────────


class FakeLLMConnector:
    """Minimal LLM connector that returns a preset JSON string."""

    def __init__(self, response: str):
        self._response = response

    async def acomplete(self, *args, **kwargs):  # noqa: ARG002
        return self._response

    async def __call__(self, *args, **kwargs):  # noqa: ARG002
        return self._response


class TestVerifyReport:
    @pytest.mark.asyncio
    async def test_pass_through_on_empty_report(self):
        result = await verify_report(
            llm_connector=FakeLLMConnector("{}"),
            question="What is EBITDA?",
            report_text="",
            evidence_pack="source 1: ...",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_pass_through_on_llm_error(self):
        """If the LLM call fails, verifier should not block the pipeline."""

        class FailingConnector:
            async def acomplete(self, *args, **kwargs):
                raise RuntimeError("LLM down")

            async def __call__(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        result = await verify_report(
            llm_connector=FailingConnector(),
            question="What is X?",
            report_text="X is 42.",
            evidence_pack="source 1: X is 42.",
        )
        assert result.passed is True
        assert result.thinking == "verifier_error"
