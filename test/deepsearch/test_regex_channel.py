import pytest

from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import LocateTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope


class _StubDense:
    def invoke(self, query, *, k, owner_id, with_score=False):  # noqa: ARG002
        assert with_score is True
        return []


class _StubBM25:
    def __init__(self, chunks_by_owner):
        self._chunks_by_owner = dict(chunks_by_owner or {})

    def invoke(self, query, *, k, owner_id, with_score=False, use_phrase_query=False, filters=None):  # noqa: ARG002
        assert with_score is True
        return list(self._chunks_by_owner.get(owner_id, []))[: int(k)]


@pytest.mark.asyncio
async def test_regex_channel_scores_by_match_count(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    chunks = [
        Chunk(id="c1", owner_id="owner-1", content="EBITDA EBITDA grew", metadata={"source_file_id": "f1", "score": 1.0}),
        Chunk(id="c2", owner_id="owner-1", content="Revenue grew", metadata={"source_file_id": "f2", "score": 1.0}),
    ]
    tool = LocateTool(dense_retriever=_StubDense(), bm25_retriever=_StubBM25({"owner-1": chunks}), llm_connector=None)
    req = ToolRunRequest(
        question="EBITDA",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"]},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool._run_regex(request=req, query="EBITDA", top_k=10, file_scope=None, regex_patterns=["EBITDA"])
    assert result.channel == "regex"
    assert len(result.evidences) >= 1
    assert result.evidences[0].chunk_id == "c1"
    assert result.evidences[0].score >= 2.0


@pytest.mark.asyncio
async def test_regex_invalid_pattern_skipped(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    chunks = [Chunk(id="c1", owner_id="owner-1", content="some text", metadata={"source_file_id": "f1", "score": 1.0})]
    tool = LocateTool(dense_retriever=_StubDense(), bm25_retriever=_StubBM25({"owner-1": chunks}), llm_connector=None)
    req = ToolRunRequest(
        question="test",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"]},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool._run_regex(
        request=req, query="test", top_k=10, file_scope=None, regex_patterns=["[invalid", "some"]
    )
    assert len(result.diagnostics.get("invalid_patterns", [])) == 1
    assert len(result.evidences) >= 1


@pytest.mark.asyncio
async def test_regex_case_insensitive(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    chunks = [
        Chunk(id="c1", owner_id="owner-1", content="The ebitda was high", metadata={"source_file_id": "f1", "score": 1.0})
    ]
    tool = LocateTool(dense_retriever=_StubDense(), bm25_retriever=_StubBM25({"owner-1": chunks}), llm_connector=None)
    req = ToolRunRequest(
        question="EBITDA",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"]},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool._run_regex(request=req, query="EBITDA", top_k=10, file_scope=None, regex_patterns=["EBITDA"])
    assert len(result.evidences) == 1


@pytest.mark.asyncio
async def test_regex_empty_patterns_skip(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    tool = LocateTool(dense_retriever=_StubDense(), bm25_retriever=_StubBM25({"owner-1": []}), llm_connector=None)
    req = ToolRunRequest(
        question="test",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"owner_ids": ["owner-1"]},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool._run_regex(request=req, query="test", top_k=10, file_scope=None, regex_patterns=[])
    assert result.channel == "regex"
    assert len(result.evidences) == 0
    assert result.diagnostics.get("reason") == "no_patterns"

