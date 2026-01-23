import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore import ExploreTool


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
    assert "explore completed" in result.summary
    errors = result.diagnostics.get("errors") or []
    assert errors
    assert "tool_not_allowed" in errors[0]
