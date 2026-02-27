import json

import pytest

from core.deepsearch.tools.think.think import ThinkTool
from core.deepsearch.tools.base import ToolRunRequest
from config.core.deepsearch import tool_defaults


class _StubLLM:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict] = []

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.output


@pytest.mark.asyncio
async def test_think_prompt_includes_l0_digest_and_truncates_cards(monkeypatch) -> None:
    monkeypatch.setattr(tool_defaults, "THINK_CONTEXT_EVIDENCE_MAX_CARDS", 3)
    monkeypatch.setattr(tool_defaults, "THINK_EVIDENCE_L0_DIGEST_ENABLED", True)
    monkeypatch.setattr(tool_defaults, "THINK_EVIDENCE_L0_DIGEST_MAX_FILES", 3)
    monkeypatch.setattr(tool_defaults, "THINK_EVIDENCE_L0_DIGEST_MAX_RANGES_PER_FILE", 4)
    monkeypatch.setattr(tool_defaults, "THINK_CURRENT_PLAN_MAX_ITEMS", 2)

    llm = _StubLLM(
        json.dumps(
            {
                "reasoning": "ok",
                "tool_calls": [],
                "plan": [],
            },
            ensure_ascii=False,
        )
    )
    tool = ThinkTool(llm_connector=llm, json_repair_attempts=0)

    cards = []
    for page in range(10):
        cards.append(
            {
                "chunk_id": f"ev{page}",
                "source": "read.pages",
                "kind": "primary",
                "provenance": {
                    "source_file_id": "file-1",
                    "page_start": page,
                    "page_end": page,
                    "node_types": {"list": 1} if page == 9 else {},
                    "metadata": {"filename": "doc1.md"},
                },
            }
        )

    req = ToolRunRequest(
        question="q",
        plan_step="p1",
        context_evidences=cards,
        adapter=None,
        access_scope=None,
        extra={"current_plan": [{"text": "a", "checked": False}, {"text": "b", "checked": True}, {"text": "c", "checked": True}]},
        graph_context=None,
        coverage_metrics={},
    )
    await tool.run(req)

    assert llm.calls, "expected an LLM call"
    payload = json.loads(llm.calls[-1]["messages"][-1]["content"])
    assert payload.get("context_evidences_total") == 10
    assert isinstance(payload.get("evidence_l0_digest"), dict)
    assert len(payload.get("context_evidences") or []) == 3
    # Plan is compacted but should retain the unchecked item.
    plan = payload.get("current_plan") or []
    assert any((isinstance(item, dict) and item.get("checked") is False) for item in plan)

