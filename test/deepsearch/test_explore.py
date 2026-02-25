import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.deepsearch import GraphQueryContext
from core.deepsearch.tools import ToolDescriptor, ToolResult, ToolRunRequest
from core.deepsearch.tools.explore import ExploreTool
from core.deepsearch.tools.governance_tags import EVIDENCE_PRIMARY, SCOPE_OWNER


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


@pytest.mark.asyncio
async def test_explore_runs_actions() -> None:
    tool = ExploreTool(
        tool_overrides={
            "locate": DummyTool("locate", "locate ok", "l1"),
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
                {"id": "a1", "tool": "locate", "args": {"top_k": 2}},
                {"id": "a2", "tool": "graph.ops", "args": {"mode": "template", "template": "path_exists", "template_args": {"source": "A", "target": "B"}}},
            ]
        },
        graph_context=GraphQueryContext(
            adapter_name="stub",
            question="Q",
            metadata={"file_scope": {"file_ids": ["11111111-1111-1111-1111-111111111111"], "source": "test"}},
        ),
    )
    result = await tool.run(request)
    assert len(result.evidences) == 2
    # Summary is a JSON envelope for LLM visibility.
    assert "\"thinking\"" in result.summary
    assert "\"answer\"" in result.summary
    assert result.diagnostics["actions"][0]["tool"] == "locate"
    assert result.diagnostics["actions"][1]["tool"] == "graph.ops"


@pytest.mark.asyncio
async def test_explore_rejects_read_chunk() -> None:
    tool = ExploreTool()
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_02",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={
            "actions": [
                {"tool": "read.chunk", "args": {"chunk_ids": ["c1"], "goal": "validate evidence", "file_id": "f1"}},
            ],
        },
    )
    result = await tool.run(request)
    assert len(result.evidences) == 0
    action = result.diagnostics["actions"][0]
    assert action["tool"] == "read.chunk"
    assert action["status"] == "failed"
    assert action["diagnostics"]["reason"] == "tool_not_allowed"


@pytest.mark.asyncio
async def test_explore_rejects_read_neighbors() -> None:
    tool = ExploreTool()
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_03",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={
            "actions": [
                {"tool": "read.neighbors", "args": {"chunk_id": "c1", "before": 2, "after": 2}},
            ],
        },
    )
    result = await tool.run(request)
    assert len(result.evidences) == 0
    action = result.diagnostics["actions"][0]
    assert action["tool"] == "read.neighbors"
    assert action["status"] == "failed"
    assert action["diagnostics"]["reason"] == "tool_not_allowed"


@pytest.mark.asyncio
async def test_explore_allows_locate_without_file_id() -> None:
    tool = ExploreTool(
        tool_overrides={
            "locate": DummyTool("locate", "locate ok", "l1"),
        }
    )
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_04",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={
            "actions": [
                {"id": "a1", "tool": "locate", "args": {"top_k": 3}},
            ]
        },
    )
    result = await tool.run(request)
    assert len(result.evidences) == 1
    assert result.diagnostics["actions"][0]["tool"] == "locate"
    gate = result.diagnostics.get("file_routing_gate") or {}
    assert gate == {}
