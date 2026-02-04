import pytest

from encapsulation.data_model.deepsearch import GraphQueryContext
from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import (
    SearchGlobalTool,
    SearchScopedTool,
    SearchFaissTool,
    SearchBM25Tool,
    SearchGraphChunkTool,
    ToolRunRequest,
)
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
    dense = _StubDenseRetriever([_chunk("c1", file_id="11111111-1111-1111-1111-111111111111")])
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
async def test_search_scoped_rejects_non_uuid_file_id() -> None:
    dense = _StubDenseRetriever([_chunk("c1", file_id="11111111-1111-1111-1111-111111111111")])
    tool = SearchScopedTool(llm_connector=None, dense_retriever=dense, bm25_retriever=None)
    req = ToolRunRequest(
        question="q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"channels": ["faiss"], "top_k": 3, "file_id": "images/ba35de39941286bbe5e3e4dbd4a9fc51.jpg"},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "invalid file_id" in result.summary.lower()
    assert result.diagnostics.get("reason") == "invalid_file_id_format"
    assert not result.evidences


@pytest.mark.asyncio
async def test_search_global_ignores_inherited_file_scope() -> None:
    dense = _StubDenseRetriever([_chunk("c1", file_id="11111111-1111-1111-1111-111111111111")])
    tool = SearchGlobalTool(llm_connector=None, dense_retriever=dense, bm25_retriever=None)
    ctx = GraphQueryContext(
        adapter_name="stub",
        question="q",
        access_scope=GraphAccessScope(scope_id="owner-1"),
        metadata={"file_scope": {"file_ids": ["22222222-2222-2222-2222-222222222222"], "source": "test"}},
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_cls",
    [SearchFaissTool, SearchBM25Tool, SearchGraphChunkTool],
)
async def test_scoped_channel_rejects_non_uuid_file_id(tool_cls) -> None:
    tool = tool_cls(llm_connector=None, dense_retriever=None, bm25_retriever=None)
    req = ToolRunRequest(
        question="q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"channels": ["faiss"], "top_k": 3, "file_id": "images/ba35de39941286bbe5e3e4dbd4a9fc51.jpg"},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "invalid file_id" in (result.summary or "").lower()
    assert result.diagnostics.get("reason") == "invalid_file_id_format"
