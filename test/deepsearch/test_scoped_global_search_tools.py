import pytest

from encapsulation.data_model.deepsearch import GraphQueryContext
from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import SearchGlobalTool, SearchScopedTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope


class _StubDenseRetriever:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def invoke(self, query, k, owner_id, with_score=True):  # noqa: ARG002
        return list(self._chunks)[:k]


def _chunk(chunk_id: str, *, file_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        content=f"content for {chunk_id}",
        metadata={"score": 0.9, "source_file_id": file_id, "file_name": f"{chunk_id}.md"},
    )


@pytest.mark.asyncio
async def test_search_scoped_requires_file_scope() -> None:
    dense = _StubDenseRetriever([_chunk("c1", file_id="file-1")])
    tool = SearchScopedTool(llm_connector=None, dense_retriever=dense, bm25_retriever=None)
    req = ToolRunRequest(
        question="q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"channels": ["faiss"], "top_k": 3},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "search.scoped skipped" in result.summary
    assert result.diagnostics.get("reason") == "missing_file_scope"
    assert not result.evidences


@pytest.mark.asyncio
async def test_search_global_ignores_inherited_file_scope() -> None:
    dense = _StubDenseRetriever([_chunk("c1", file_id="file-1")])
    tool = SearchGlobalTool(llm_connector=None, dense_retriever=dense, bm25_retriever=None)
    ctx = GraphQueryContext(
        adapter_name="stub",
        question="q",
        access_scope=GraphAccessScope(scope_id="owner-1"),
        metadata={"file_scope": {"file_ids": ["file-2"], "source": "test"}},
    )
    req = ToolRunRequest(
        question="q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"channels": ["faiss"], "top_k": 3},
        graph_context=ctx,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    assert result.evidences, "global search should not inherit file_scope and drop results"
    assert result.diagnostics.get("search_mode") == "global"
    assert "global_search_may_introduce_cross_doc_noise" in (result.diagnostics.get("risk") or "")
