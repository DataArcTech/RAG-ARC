import pytest

from core.deepsearch.report.consistency_checker import ConsistencyChecker
from core.deepsearch.report.quality_gate import DeepSearchQualityGate


@pytest.mark.asyncio
async def test_consistency_checker_retries_on_invalid_json_then_succeeds(monkeypatch):
    responses = [
        '```json\n{"is_consistent": true, "confidence": 0.9, "issues": [{"issue_type":"checker_error","location":"summary","description":"bad\nnewline"}]}\n```',
        '{"is_consistent": true, "confidence": 0.9, "issues": []}',
    ]
    calls: list[list[dict]] = []

    async def fake_call_llm_async(llm, messages, **kwargs):  # noqa: ARG001
        calls.append(list(messages))
        return responses.pop(0)

    import core.deepsearch.report.consistency_checker as module

    monkeypatch.setattr(module, "call_llm_async", fake_call_llm_async)

    checker = ConsistencyChecker(llm_connector=object(), temperature=0.0, max_retries=2)
    result = await checker.check(
        question="What is the guaranteed preferential interest rate?",
        report_markdown="Guaranteed preferential interest rate is 3%.[chunk_001]",
        structured_report={
            "summary": "Guaranteed preferential interest rate is 3%.[chunk_001]",
            "citations": [{"evidence_id": "chunk_001", "used_for": "support"}],
        },
        evidences=[{"chunk_id": "chunk_001", "content": "Guaranteed Preferential Interest Rate: 3%"}],
    )

    assert result.is_consistent is True
    # Parsing now auto-repairs unescaped newlines inside JSON strings, so no LLM retry is required.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_quality_gate_llm_judge_retries_on_invalid_json_then_succeeds(monkeypatch):
    responses = [
        '```json\n{"pass": true, "overall": 0.9, "scores": {"factual_accuracy": 1.0, "citation_accuracy": 1.0, "completeness": 0.8, "source_quality": 0.8}, "reasons": ["ok\nbad"], "missing_topics": [], "missing_claims": [], "next_actions": []}\n```',
        '{"pass": true, "overall": 0.9, "scores": {"factual_accuracy": 1.0, "citation_accuracy": 1.0, "completeness": 0.8, "source_quality": 0.8}, "reasons": ["ok"], "missing_topics": [], "missing_claims": [], "next_actions": []}',
    ]
    calls: list[list[dict]] = []

    async def fake_call_llm_async(llm, messages, **kwargs):  # noqa: ARG001
        calls.append(list(messages))
        return responses.pop(0)

    import core.deepsearch.report.quality_gate as module

    monkeypatch.setattr(module, "call_llm_async", fake_call_llm_async)

    gate = DeepSearchQualityGate(
        llm_connector=object(),
        config={
            "enabled": True,
            "max_rounds": 2,
            "min_citation_sentence_coverage": 0.6,
            "require_consistency": False,
            "max_uncited_sentences": 3,
            "max_actions": 4,
            "enable_llm_judge": True,
            "judge_temperature": 0.0,
            "judge_max_retries": 2,
            "judge_max_evidence_items": 5,
            "judge_max_evidence_chars": 200,
        },
    )
    result = await gate.evaluate(
        question="What is the guaranteed preferential interest rate?",
        structured_report={"summary": "Guaranteed preferential interest rate is 3%.[chunk_001]"},
        evidences=[{"chunk_id": "chunk_001", "content": "Guaranteed Preferential Interest Rate: 3%"}],
    )

    assert result.judge is not None
    assert result.judge.passed is True
    # Parsing now auto-repairs unescaped newlines inside JSON strings, so no LLM retry is required.
    assert len(calls) == 1
