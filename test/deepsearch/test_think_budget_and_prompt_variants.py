import pytest

from core.deepsearch.tools.think.think import ThinkTool
from core.prompts.deepsearch import THINK_TOOL_SYSTEM_PROMPT_EN


def test_think_builds_budget_status_with_phase() -> None:
    tool = ThinkTool(llm_connector=None)
    status = tool._build_budget_status(  # type: ignore[attr-defined]
        {
            "max_calls_total": 10,
            "used_calls": 8,
            "remaining_calls": 2,
        }
    )
    assert isinstance(status, dict)
    assert status["max_calls_total"] == 10
    assert status["used_calls"] == 8
    assert status["remaining_calls"] == 2
    assert 0.0 <= float(status["remaining_ratio"]) <= 1.0
    assert status["phase"] in {"ok", "low", "critical"}


def test_think_prompt_always_returns_default() -> None:
    """After gate removal, _select_system_prompt always returns the default prompt."""
    tool = ThinkTool(llm_connector=None)
    for prev in [
        [],
        [{"status": "failed", "failure_reason": "missing_primary_page_evidence"}],
        [{"status": "ok"}],
    ]:
        selected = tool._select_system_prompt(  # type: ignore[attr-defined]
            extra={},
            previous_tool_call_results=prev,
        )
        assert selected == THINK_TOOL_SYSTEM_PROMPT_EN
