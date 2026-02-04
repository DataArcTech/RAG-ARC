import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.tree_tools import (
    TreeRootTool,
    TreeChildrenTool,
    TreeNodeTool,
    TreeOpenTool,
)
from core.graph_adapter.base import GraphAccessScope


class _FakeAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id and file_id != "11111111-1111-1111-1111-111111111111":
            return []
        if "HAS_CHUNK" in cypher:
            return [
                {"section_id": "sec-1", "metadata": {"semantic_unit_type": "image"}},
                {"section_id": "sec-1", "metadata": {"semantic_unit_type": "table"}},
            ]
        if "count(t) AS node_count" in cypher:
            return [
                {"section_id": "sec-1", "node_count": 3, "token_count": 4200},
            ]
        if "MATCH (s:Section" in cypher and "HAS_CHILD" in cypher and "RETURN t.node_id" in cypher:
            return [
                {
                    "node_id": "file:table:1",
                    "node_type": "table",
                    "page_start": 2,
                    "page_end": 2,
                    "summary": "Table: Fees",
                    "section_id": "sec-1",
                    "section_path": "Intro",
                    "file_id": file_id,
                    "resource_urls": ["images/1.png"],
                    "token_count": 200,
                },
                {
                    "node_id": "file:image:2",
                    "node_type": "image",
                    "page_start": 3,
                    "page_end": 3,
                    "summary": "Diagram",
                    "section_id": "sec-1",
                    "section_path": "Intro",
                    "file_id": file_id,
                    "resource_urls": ["images/2.png"],
                    "token_count": 120,
                },
            ]
        if "MATCH (t:TreeNode {node_id:" in cypher:
            return [
                {
                    "node_id": params.get("node_id"),
                    "node_type": "image",
                    "page_start": 4,
                    "page_end": 4,
                    "summary": "Figure 1",
                    "section_id": "sec-1",
                    "section_path": "Intro",
                    "file_id": file_id,
                    "resource_urls": ["images/fig1.png"],
                    "token_count": 80,
                }
            ]
        # Section rows returned directly from the Section index.
        return [
            {
                "section_id": "sec-1",
                "section_path": "Intro",
                "section_title": "Intro",
                "section_level": 1,
                "page_start": 0,
                "page_end": 4,
                "section_parent_id": None,
            },
            {
                "section_id": "sec-2",
                "section_path": "Intro > Safety",
                "section_title": "Safety",
                "section_level": 2,
                "page_start": 1,
                "page_end": 2,
                "section_parent_id": "sec-1",
            },
        ]


@pytest.mark.asyncio
async def test_tree_root_lists_top_sections() -> None:
    tool = TreeRootTool()
    req = ToolRunRequest(
        question="root",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "max_roots": 5},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "tree.root returned top-level sections" in result.summary
    assert "section_id=sec-1" in result.summary


@pytest.mark.asyncio
async def test_tree_children_lists_nodes() -> None:
    tool = TreeChildrenTool()
    req = ToolRunRequest(
        question="children",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "section_id": "sec-1", "max_nodes": 5},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "tree.children returned TreeNodes" in result.summary
    assert "node_id=file:table:1" in result.summary


@pytest.mark.asyncio
async def test_tree_node_returns_metadata() -> None:
    tool = TreeNodeTool()
    req = ToolRunRequest(
        question="node",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"node_id": "file:image:1"},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "tree.node" in result.summary
    assert "node_id=file:image:1" in result.summary


@pytest.mark.asyncio
async def test_tree_open_returns_summary() -> None:
    tool = TreeOpenTool()
    req = ToolRunRequest(
        question="open",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"node_id": "file:image:1"},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "tree.open" in result.summary
    assert "resource_urls=1" in result.summary


@pytest.mark.asyncio
async def test_tree_open_accepts_section_id_fallback() -> None:
    tool = TreeOpenTool()
    req = ToolRunRequest(
        question="open",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "section_id": "sec-1"},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "tree.open" in result.summary
    assert "section_id=sec-1" in result.summary
