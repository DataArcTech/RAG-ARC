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
async def test_think_tool_normalizes_function_call_style_tool_calls_without_repair() -> None:
    payload = {
        "reasoning": "need to explore next",
        "tool_calls": [{"function": "locate", "arguments": {"query": "section 5.2 table"}}],
        "plan": [{"text": "Locate section 5.2 table", "checked": False}],
    }
    llm = _StubLLM([json.dumps(payload)])
    tool = ThinkTool(llm_connector=llm, json_repair_attempts=1, json_repair_temperature=0.0)
    req = ToolRunRequest(
        question="q",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={},
    )

    result = await tool.run(req)
    assert len(llm.calls) == 1  # no JSON repair call needed
    assert result.think_notes
    meta = result.think_notes[0].metadata
    calls = meta.get("tool_calls")
    assert isinstance(calls, list) and calls
    assert calls[0]["tool_name"] == "locate"
    assert calls[0]["tool_args"] == {"query": "section 5.2 table"}
    assert isinstance(calls[0]["rationale"], str) and calls[0]["rationale"]

