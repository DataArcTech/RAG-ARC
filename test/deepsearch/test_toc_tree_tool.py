import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.toc_tree import TocTreeTool
from core.graph_adapter.base import GraphAccessScope


class _FakeAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id != "11111111-1111-1111-1111-111111111111":
            return []
        if "HAS_CHUNK" in cypher:
            return [
                {"section_id": "sec-2", "metadata": {"semantic_unit_type": "table"}},
                {"section_id": "sec-2", "metadata": {"semantic_unit_type": "table"}},
                {"section_id": "sec-3", "metadata": {"semantic_unit_type": "image"}},
            ]
        # Section rows returned directly from the Section index.
        return [
            {
                "section_id": "sec-1",
                "section_path": "Intro",
                "section_title": "Intro",
                "section_level": 1,
                "page_start": 0,
                "page_end": 0,
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
            {
                "section_id": "sec-3",
                "section_path": "Intro > Safety > Warnings",
                "section_title": "Warnings",
                "section_level": 3,
                "page_start": 2,
                "page_end": 2,
                "section_parent_id": "sec-2",
            },
            {
                "section_id": "sec-4",
                "section_path": "Procedure",
                "section_title": "Procedure",
                "section_level": 1,
                "page_start": 3,
                "page_end": 5,
                "section_parent_id": None,
            },
        ]


@pytest.mark.asyncio
async def test_toc_tree_builds_readable_tree() -> None:
    tool = TocTreeTool()
    req = ToolRunRequest(
        question="toc",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "max_depth": 4, "max_nodes": 50},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "toc.tree returned section tree" in result.summary
    assert "section_id=sec-2" in result.summary
    assert "table=" in result.summary
    assert "Intro" in result.summary
    assert "Warnings" in result.summary
    assert result.diagnostics.get("file_id") == "11111111-1111-1111-1111-111111111111"
