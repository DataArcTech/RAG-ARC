import json

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext
from core.deepsearch.tools.check import LogicCheckTool
from core.deepsearch.tools.base import ToolRunRequest


class _StubLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, **kwargs):
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_logic_check_adds_deterministic_issue_for_computable_without_code() -> None:
    llm = _StubLLM(
        [
            json.dumps(
                {
                    "summary": "Looks consistent.",
                    "ok": True,
                    "issues": [],
                }
            )
        ]
    )
    tool = LogicCheckTool(llm_connector=llm)
    context = GraphQueryContext(
        adapter_name="stub",
        metadata={"question_classification": {"is_computable": True}},
    )
    evidences = [
        EvidenceChunk(
            chunk_id="ev1",
            source="search",
            content="placeholder",
            kind="primary",
            provenance={},
        )
    ]
    payload = ToolRunRequest(
        question="Compute the ratio.",
        plan_step="plan_01",
        context_evidences=evidences,
        adapter=None,
        access_scope=None,
        graph_context=context,
        extra={
            "runtime_snapshot": {
                "tool_names": ["search"],
                "evidence_ids": ["ev1"],
            }
        },
    )
    result = await tool.run(payload)
    assert result.summary == "Looks consistent."
    assert result.evidences
    assert result.evidences[0].source == "logic.check"
    assert result.evidences[0].kind == "derived"
    issues = result.diagnostics.get("issues") or []
    issue_types = {item.get("issue_type") for item in issues if isinstance(item, dict)}
    assert "computable_without_code" in issue_types
