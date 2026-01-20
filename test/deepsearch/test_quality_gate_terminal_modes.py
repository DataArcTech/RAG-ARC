import pytest

from core.deepsearch.report.quality_gate import DeepSearchQualityGate


@pytest.mark.asyncio
async def test_quality_gate_allows_terminal_no_evidence_mode_without_llm():
    gate = DeepSearchQualityGate(
        llm_connector=None,
        config={
            "enabled": True,
            "max_rounds": 2,
            "min_citation_sentence_coverage": 0.8,
            "require_consistency": False,
            "max_uncited_sentences": 6,
            "max_actions": 6,
            "enable_llm_judge": True,
            "judge_temperature": 0.0,
            "judge_max_retries": 1,
            "judge_max_evidence_items": 5,
            "judge_max_evidence_chars": 200,
        },
    )

    result = await gate.evaluate(
        question="Only use 《missing》",
        structured_report={"generation": {"mode": "deterministic_no_evidence"}, "summary": "No evidence."},
        evidences=[],
    )

    assert result.enabled is True
    assert result.passed is True
    assert result.should_iterate is False
    assert result.actions == []
    assert result.diagnostics.get("terminal_mode") == "deterministic_no_evidence"
