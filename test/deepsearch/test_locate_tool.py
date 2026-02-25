import pytest

from encapsulation.data_model.deepsearch import GraphQueryContext
from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import LocateTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope


class _StubDenseRetriever:
    def __init__(self, by_owner):
        self._by_owner = dict(by_owner or {})

    def invoke(self, query, *, k, owner_id, with_score=False):  # noqa: ARG002
        assert with_score is True
        return list(self._by_owner.get(owner_id, []))[: int(k)]


class _StubBM25Retriever:
    def __init__(self, by_owner):
        self._by_owner = dict(by_owner or {})

    def invoke(self, query, *, k, owner_id, with_score=False, use_phrase_query=False, filters=None):  # noqa: ARG002
        assert with_score is True
        return list(self._by_owner.get(owner_id, []))[: int(k)]


@pytest.mark.asyncio
async def test_locate_aggregates_by_file_id(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    dense = _StubDenseRetriever(
        {
            "owner-1": [
                Chunk(
                    id="c1",
                    owner_id="owner-1",
                    content="hit one",
                    metadata={"source_file_id": "file-1", "score": 0.9, "filename": "a.pdf"},
                ),
                Chunk(
                    id="c2",
                    owner_id="owner-1",
                    content="hit two",
                    metadata={"source_file_id": "file-2", "score": 0.8, "filename": "b.pdf"},
                ),
            ],
        }
    )
    bm25 = _StubBM25Retriever(
        {
            "owner-1": [
                Chunk(
                    id="c3",
                    owner_id="owner-1",
                    content="bm25 hit",
                    metadata={"source_file_id": "file-2", "score": 3.0, "filename": "b.pdf"},
                ),
            ],
        }
    )
    tool = LocateTool(dense_retriever=dense, bm25_retriever=bm25, llm_connector=None)
    req = ToolRunRequest(
        question="find docs",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"], "top_k": 5},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    rows = result.diagnostics.get("results") or []
    file_ids = [row["file_id"] for row in rows]
    assert "file-2" in file_ids
    assert "file-1" in file_ids
    assert file_ids[0] == "file-2"


@pytest.mark.asyncio
async def test_locate_rejects_unknown_owner(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    tool = LocateTool(dense_retriever=_StubDenseRetriever({}), bm25_retriever=_StubBM25Retriever({}), llm_connector=None)
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
    assert "missing" in result.summary.lower() or "skipped" in result.summary.lower()


@pytest.mark.asyncio
async def test_locate_uses_query_spec_regex(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    bm25 = _StubBM25Retriever(
        {
            "owner-1": [
                Chunk(
                    id="c1",
                    owner_id="owner-1",
                    content="EBITDA was $1.5B in FY2023",
                    metadata={"source_file_id": "file-1", "score": 2.0, "filename": "a.pdf"},
                ),
                Chunk(
                    id="c2",
                    owner_id="owner-1",
                    content="Revenue grew 10%",
                    metadata={"source_file_id": "file-2", "score": 1.5, "filename": "b.pdf"},
                ),
            ],
        }
    )
    dense = _StubDenseRetriever({"owner-1": []})
    tool = LocateTool(dense_retriever=dense, bm25_retriever=bm25, llm_connector=None)

    ctx = GraphQueryContext(
        adapter_name="stub",
        question="what is EBITDA for FY2023?",
        access_scope=GraphAccessScope(scope_id="owner-1"),
        metadata={
            "query_spec": {
                "bm25_terms": ["EBITDA", "FY2023"],
                "regex_patterns": ["EBITDA"],
                "hyde_query": "The EBITDA for FY2023 was approximately...",
            }
        },
    )
    req = ToolRunRequest(
        question="what is EBITDA for FY2023?",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"], "top_k": 5},
        graph_context=ctx,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    diag = result.diagnostics
    assert diag.get("query_spec_present") is True
    assert diag.get("regex_patterns") == ["EBITDA"]
    rows = diag.get("results") or []
    assert rows, "expected locate results"
    assert rows[0]["file_id"] == "file-1"
