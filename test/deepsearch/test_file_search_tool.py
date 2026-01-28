import pytest

from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.search.file_search import FileSearchTool
from core.graph_adapter.base import GraphAccessScope


class _StubPageIndexRetriever:
    def __init__(self, by_owner):
        self._by_owner = by_owner

    def retrieve_doc_chunks(self, query, *, owner_id, k_final=None, k_candidates=None):  # noqa: ARG002
        return list(self._by_owner.get(owner_id, []))[: (k_final or 5)]


@pytest.mark.asyncio
async def test_file_search_returns_doc_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAGEINDEX_ENABLED", "1")
    monkeypatch.setenv("DOC_ROUTING_ENABLED", "1")
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")

    stub = _StubPageIndexRetriever(
        {
            "owner-1": [
                Chunk(
                    id="file-1",
                    owner_id="owner-1",
                    content="Doc One\nThis is description one.",
                    metadata={"source_file_id": "file-1", "filename": "a.pdf", "score": 0.3},
                )
            ],
            "share-1": [
                Chunk(
                    id="file-2",
                    owner_id="share-1",
                    content="Doc Two\nThis is description two.",
                    metadata={"source_file_id": "file-2", "filename": "b.pdf", "score": 0.6},
                )
            ],
        }
    )

    tool = FileSearchTool(pageindex_retriever=stub)
    req = ToolRunRequest(
        question="find docs",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1", "share-1"], "top_k": 5},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    assert "desc:" in result.summary
    rows = result.diagnostics.get("results") or []
    assert [row["file_id"] for row in rows] == ["file-2", "file-1"]
    assert rows[0]["doc_description"] == "This is description two."


@pytest.mark.asyncio
async def test_file_search_rejects_unknown_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAGEINDEX_ENABLED", "1")
    monkeypatch.setenv("DOC_ROUTING_ENABLED", "1")
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")

    stub = _StubPageIndexRetriever({"owner-1": []})
    tool = FileSearchTool(pageindex_retriever=stub)
    req = ToolRunRequest(
        question="find docs",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["other-1"]},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    visibility = result.diagnostics.get("owner_visibility") or {}
    assert visibility.get("owner_ids_rejected") == ["other-1"]
