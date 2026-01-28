import pytest

from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.search.section_search import SectionSearchTool
from core.graph_adapter.base import GraphAccessScope


class _StubPageIndexRetriever:
    def __init__(self, by_owner):
        self._by_owner = by_owner

    def retrieve_sections(self, query, *, owner_id, file_ids=None):  # noqa: ARG002
        items = list(self._by_owner.get(owner_id, []))
        if file_ids:
            allowed = set(file_ids)
            items = [c for c in items if (c.metadata or {}).get("source_file_id") in allowed]
        return items


@pytest.mark.asyncio
async def test_section_search_returns_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAGEINDEX_ENABLED", "1")
    monkeypatch.setenv("SECTION_INDEX_ENABLED", "1")
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")

    stub = _StubPageIndexRetriever(
        {
            "owner-1": [
                Chunk(
                    id="sec-1",
                    owner_id="owner-1",
                    content="Title\nPath\nSummary",
                    metadata={"section_id": "sec-1", "section_path": "A/B", "source_file_id": "file-1", "score": 0.2},
                )
            ],
            "share-1": [
                Chunk(
                    id="sec-2",
                    owner_id="share-1",
                    content="Title2\nPath2\nSummary2",
                    metadata={"section_id": "sec-2", "section_path": "C/D", "source_file_id": "file-2", "score": 0.6},
                )
            ],
        }
    )

    tool = SectionSearchTool(pageindex_retriever=stub)
    req = ToolRunRequest(
        question="find sections",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1", "share-1"], "top_k": 5},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    rows = result.diagnostics.get("results") or []
    assert [row["section_id"] for row in rows] == ["sec-2", "sec-1"]


@pytest.mark.asyncio
async def test_section_search_respects_file_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAGEINDEX_ENABLED", "1")
    monkeypatch.setenv("SECTION_INDEX_ENABLED", "1")

    stub = _StubPageIndexRetriever(
        {
            "owner-1": [
                Chunk(id="sec-1", owner_id="owner-1", content="x", metadata={"section_id": "sec-1", "source_file_id": "file-1"}),
                Chunk(id="sec-2", owner_id="owner-1", content="y", metadata={"section_id": "sec-2", "source_file_id": "file-2"}),
            ]
        }
    )

    tool = SectionSearchTool(pageindex_retriever=stub)
    req = ToolRunRequest(
        question="find sections",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "file-2"},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    rows = result.diagnostics.get("results") or []
    assert [row["section_id"] for row in rows] == ["sec-2"]
