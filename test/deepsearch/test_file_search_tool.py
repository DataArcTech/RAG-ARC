import pytest

from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.search.file_search import FileSearchTool
from core.graph_adapter.base import GraphAccessScope


class _StubDenseRetriever:
    def __init__(self, by_owner):  # noqa: ANN001
        self._by_owner = dict(by_owner or {})

    def invoke(self, query, *, k, owner_id, with_score=False):  # noqa: ARG002
        assert with_score is True
        return list(self._by_owner.get(owner_id, []))[: int(k)]


class _StubBM25Retriever:
    def __init__(self, by_owner):  # noqa: ANN001
        self._by_owner = dict(by_owner or {})

    def invoke(self, query, *, k, owner_id, with_score=False, use_phrase_query=False, filters=None):  # noqa: ARG002
        assert with_score is True
        return list(self._by_owner.get(owner_id, []))[: int(k)]


class _StubLLM:
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: D401, ARG002
        self.calls += 1
        return self._response


@pytest.mark.asyncio
async def test_file_search_aggregates_by_file_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")

    dense = _StubDenseRetriever(
        {
            "owner-1": [
                Chunk(id="c1", owner_id="owner-1", content="hit one", metadata={"source_file_id": "file-1", "score": 0.9, "filename": "a.pdf"}),
                Chunk(id="c2", owner_id="owner-1", content="hit two", metadata={"source_file_id": "file-2", "score": 0.8, "filename": "b.pdf"}),
            ],
            "share-1": [
                Chunk(id="c3", owner_id="share-1", content="hit three", metadata={"source_file_id": "file-2", "score": 0.7, "filename": "b.pdf"}),
            ],
        }
    )
    bm25 = _StubBM25Retriever(
        {
            "owner-1": [
                Chunk(id="c4", owner_id="owner-1", content="bm25 hit", metadata={"source_file_id": "file-2", "score": 3.0, "filename": "b.pdf"}),
            ],
            "share-1": [],
        }
    )

    tool = FileSearchTool(dense_retriever=dense, bm25_retriever=bm25, llm_connector=None)
    req = ToolRunRequest(
        question="find docs",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1", "share-1"], "top_k": 5, "channels": ["faiss", "bm25"]},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    rows = result.diagnostics.get("results") or []
    assert [row["file_id"] for row in rows] == ["file-2", "file-1"]
    assert rows[0]["filename"] == "b.pdf"
    assert int(rows[0]["hit_count"]) >= 2


@pytest.mark.asyncio
async def test_file_search_rejects_unknown_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")

    dense = _StubDenseRetriever({"owner-1": []})
    bm25 = _StubBM25Retriever({"owner-1": []})
    tool = FileSearchTool(dense_retriever=dense, bm25_retriever=bm25, llm_connector=None)

    req = ToolRunRequest(
        question="find docs",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["other-1"], "channels": ["faiss"]},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    visibility = result.diagnostics.get("owner_visibility") or {}
    assert visibility.get("owner_ids_rejected") == ["other-1"]


@pytest.mark.asyncio
async def test_file_search_llm_rerank_changes_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")

    dense = _StubDenseRetriever(
        {
            "owner-1": [
                Chunk(id="c1", owner_id="owner-1", content="hit one", metadata={"source_file_id": "file-1", "score": 0.9, "filename": "a.pdf"}),
                Chunk(id="c2", owner_id="owner-1", content="hit two", metadata={"source_file_id": "file-2", "score": 0.8, "filename": "b.pdf"}),
            ]
        }
    )
    bm25 = _StubBM25Retriever(
        {
            "owner-1": [
                Chunk(id="c3", owner_id="owner-1", content="bm25 hit", metadata={"source_file_id": "file-2", "score": 3.0, "filename": "b.pdf"}),
            ]
        }
    )
    llm = _StubLLM('{"thinking": "User intent matches file-1.", "answer": ["file-1", "file-2"]}')
    tool = FileSearchTool(dense_retriever=dense, bm25_retriever=bm25, llm_connector=llm)
    req = ToolRunRequest(
        question="find docs",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"], "top_k": 5, "channels": ["faiss", "bm25"]},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    rows = result.diagnostics.get("results") or []
    assert [row["file_id"] for row in rows][:2] == ["file-1", "file-2"]
    assert result.diagnostics.get("llm_rerank", {}).get("ranked_file_ids") == ["file-1", "file-2"]
    assert llm.calls >= 1


@pytest.mark.asyncio
async def test_file_search_llm_rerank_accepts_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")

    dense = _StubDenseRetriever(
        {
            "owner-1": [
                Chunk(id="c1", owner_id="owner-1", content="hit one", metadata={"source_file_id": "file-1", "score": 0.9, "filename": "a.pdf"}),
                Chunk(id="c2", owner_id="owner-1", content="hit two", metadata={"source_file_id": "file-2", "score": 0.8, "filename": "b.pdf"}),
            ]
        }
    )
    bm25 = _StubBM25Retriever(
        {
            "owner-1": [
                Chunk(id="c3", owner_id="owner-1", content="bm25 hit", metadata={"source_file_id": "file-2", "score": 3.0, "filename": "b.pdf"}),
            ]
        }
    )
    llm = _StubLLM('{"thinking": "Use filenames.", "answer": ["b.pdf", "a.pdf"]}')
    tool = FileSearchTool(dense_retriever=dense, bm25_retriever=bm25, llm_connector=llm)
    req = ToolRunRequest(
        question="find docs",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"], "top_k": 5, "channels": ["faiss", "bm25"]},
        graph_context=None,
        coverage_metrics=None,
    )

    result = await tool.run(req)
    rows = result.diagnostics.get("results") or []
    assert [row["file_id"] for row in rows][:2] == ["file-2", "file-1"]
    assert llm.calls >= 1
