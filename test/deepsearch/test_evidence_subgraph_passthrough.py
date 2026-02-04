import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore import ExploreTool
from core.utils.json_extract import safe_json_loads


@pytest.mark.asyncio
async def test_explore_rejects_unknown_action_tool() -> None:
    tool = ExploreTool()
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={"actions": [{"tool": "unknown.tool", "args": {}}]},
    )

    result = await tool.run(request)
    payload = safe_json_loads(result.summary, expected="dict")
    assert payload["answer"]["ok_actions"] == 0
    assert payload["answer"]["total_actions"] == 1
    errors = result.diagnostics.get("errors") or []
    assert errors
    assert "tool_not_allowed" in errors[0]
