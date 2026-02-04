import json

import pytest

from core.deepsearch.tools import ThinkTool, ToolRunRequest


class _StubLLM:
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self._outputs:
            raise RuntimeError("stub exhausted")
        return self._outputs.pop(0)


@pytest.mark.asyncio
async def test_think_tool_repairs_invalid_json_output() -> None:
    valid = {
        "reasoning": "ok",
        "tool_calls": [],
        "plan": [],
    }
    llm = _StubLLM(["not json", json.dumps(valid)])
    tool = ThinkTool(llm_connector=llm, json_repair_attempts=1, json_repair_temperature=0.0)
    req = ToolRunRequest(
        question="q",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={},
        graph_context=None,
        coverage_metrics={"coverage_score": 0.2, "confidence_score": 0.2},
    )
    result = await tool.run(req)
    assert result.think_notes
    assert len(llm.calls) == 2
    assert llm.calls[0]["kwargs"].get("temperature") == pytest.approx(tool.temperature)
    assert llm.calls[1]["kwargs"].get("temperature") == pytest.approx(tool.json_repair_temperature)


@pytest.mark.asyncio
async def test_think_tool_raises_when_json_repair_disabled() -> None:
    llm = _StubLLM(["not json"])
    tool = ThinkTool(llm_connector=llm, json_repair_attempts=0)
    req = ToolRunRequest(
        question="q",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={},
    )
    with pytest.raises(RuntimeError, match="non-JSON"):
        await tool.run(req)


@pytest.mark.asyncio
async def test_think_tool_repairs_schema_error() -> None:
    invalid = {
        "reasoning": "ok",
        "tool_calls": [],
        "plan": [],
    }
    llm = _StubLLM([json.dumps(invalid)])
    tool = ThinkTool(llm_connector=llm, json_repair_attempts=1, json_repair_temperature=0.0)
    req = ToolRunRequest(
        question="q",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={"think_mode": "final"},
    )
    result = await tool.run(req)
    # Final-mode robustness: when the model omits `is_final` but provides no tool calls,
    # ThinkTool should safely fill `is_final=true` without triggering JSON repair.
    assert len(llm.calls) == 1
    assert result.think_notes[0].metadata.get("is_final") is True
    assert result.think_notes[0].metadata.get("schema_repair") is None
