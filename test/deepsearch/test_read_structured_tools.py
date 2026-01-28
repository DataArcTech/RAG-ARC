import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.read_structured import ReadPagesTool, ReadSectionTool
from core.graph_adapter.base import GraphAccessScope


class _FakeAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id != "file-1":
            return []
        return [
            {
                "chunk_id": "c1",
                "content": "Intro paragraph.",
                "metadata": {"chunk_index": 0, "section_id": "sec-intro", "page_start": 0, "page_end": 0},
            },
            {
                "chunk_id": "c2",
                "content": "Safety warning A.",
                "metadata": {"chunk_index": 1, "section_id": "sec-safety", "page_start": 1, "page_end": 1},
            },
            {
                "chunk_id": "c3",
                "content": "Safety warning B.",
                "metadata": {"chunk_index": 2, "section_id": "sec-safety", "page_start": 2, "page_end": 2},
            },
            {
                "chunk_id": "c4",
                "content": "Procedure step 1.",
                "metadata": {"chunk_index": 3, "section_id": "sec-proc", "page_start": 3, "page_end": 3},
            },
        ]


@pytest.mark.asyncio
async def test_read_section_returns_ordered_full_text() -> None:
    tool = ReadSectionTool()
    req = ToolRunRequest(
        question="read",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "file-1", "section_id": "sec-safety", "max_chars": 5000, "max_chunks": 20},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "read.section returned" in result.summary
    assert len(result.evidences) == 1
    evidence = result.evidences[0]
    assert evidence.source == "read.section"
    assert "Safety warning A." in evidence.content
    assert "Safety warning B." in evidence.content


@pytest.mark.asyncio
async def test_read_pages_filters_by_page_range() -> None:
    tool = ReadPagesTool()
    req = ToolRunRequest(
        question="read",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "file-1", "page_start": 1, "page_end": 2, "max_chars": 5000, "max_chunks": 20},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "read.pages returned" in result.summary
    assert len(result.evidences) == 1
    evidence = result.evidences[0]
    assert evidence.source == "read.pages"
    assert "Safety warning A." in evidence.content
    assert "Safety warning B." in evidence.content
    assert "Procedure step 1." not in evidence.content
