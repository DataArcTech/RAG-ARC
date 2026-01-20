from typing import Any, Dict, List

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.tools import ToolDescriptor, ToolResult, ToolRunRequest
from core.deepsearch.tools.fast.knowledge_base_explore import KnowledgeBaseExploreTool
from core.deepsearch.tools.governance_tags import EVIDENCE_PRIMARY, SCOPE_OWNER
from core.graph_adapter.base import GraphAccessScope


class DummyTool:
    def __init__(self, name: str, summary: str, evidence_id: str):
        self.descriptor = ToolDescriptor(
            name=name,
            channel="graph",
            description="dummy",
            strategy_tags=(EVIDENCE_PRIMARY, SCOPE_OWNER),
        )
        self._summary = summary
        self._evidence_id = evidence_id

    async def run(self, request: ToolRunRequest) -> ToolResult:
        evidence = EvidenceChunk(
            chunk_id=self._evidence_id,
            source=self.descriptor.name,
            content="dummy",
        )
        return ToolResult(summary=self._summary, evidences=[evidence], diagnostics={"dummy": True})


class FakeAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params: Dict[str, Any] | None = None, *, access_scope=None) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": "c1",
                "content": "chunk content",
                "metadata": '{"source_file_id": "f1"}',
                "source_file_id": "f1",
                "owner_id": "owner",
            }
        ]


@pytest.mark.asyncio
async def test_knowledge_base_explore_runs_actions() -> None:
    tool = KnowledgeBaseExploreTool(
        tool_overrides={
            "search": DummyTool("search", "search ok", "s1"),
            "graph.ops": DummyTool("graph.ops", "path ok", "g1"),
        }
    )
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={
            "actions": [
                {"id": "a1", "tool": "search", "args": {"top_k": 2, "channels": ["faiss"]}},
                {"id": "a2", "tool": "graph.ops", "args": {"mode": "template", "template": "path_exists", "template_args": {"source": "A", "target": "B"}}},
            ]
        },
    )
    result = await tool.run(request)
    assert len(result.evidences) == 2
    assert "knowledge_base.explore completed" in result.summary
    assert result.diagnostics["actions"][0]["tool"] == "search"
    assert result.diagnostics["actions"][1]["tool"] == "graph.ops"


@pytest.mark.asyncio
async def test_knowledge_base_explore_read_chunks() -> None:
    tool = KnowledgeBaseExploreTool()
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_02",
        context_evidences=[],
        adapter=FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner"),
        extra={
            "actions": [
                {"tool": "read_chunks", "args": {"chunk_ids": ["c1"], "goal": "validate evidence"}},
            ],
        },
    )
    result = await tool.run(request)
    assert len(result.evidences) == 1
    evidence = result.evidences[0]
    assert evidence.chunk_id == "c1"
    assert evidence.source == "knowledge_base.read_chunks"
    assert evidence.provenance.get("goal") == "validate evidence"
