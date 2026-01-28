import types

import numpy as np
import pytest

from encapsulation.data_model.schema import Chunk
from core.deepsearch.tools import SearchTool, SearchGraphChunkTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope


class _StubLLM:
    def chat(self, messages, **kwargs):  # noqa: D401, ARG002
        return '{"entities": []}'


class _StubDenseRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def invoke(self, query, k, owner_id, with_score=True):  # noqa: ARG002
        return list(self._chunks)[:k]


class _StubBM25Retriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def invoke(self, query, k, owner_id, with_score=True, use_phrase_query=False, filters=None):  # noqa: ARG002
        return list(self._chunks)[:k]


class _StubGraphRetriever:
    def __init__(self):
        self.config = types.SimpleNamespace(
            fact_retrieval_top_k=2,
            seed_entities_from_entity_nn_enabled=False,
            enable_pruning=False,
            dense_mix_in_top_k=0,
        )
        self.passage_node_keys = ["chunk-1", "chunk-2", "chunk-3"]

    def _dense_passage_retrieval_scores(self, query):  # noqa: ARG002
        return np.array([0.8, 0.1, 0.6])

    def _get_fact_scores_faiss(self, query, owner_id, query_doc_scores=None):  # noqa: ARG002
        return np.array([0.9]), ["fact-1"]

    def _get_facts_by_indices(self, indices, fact_ids, owner_id):  # noqa: ARG002
        return [{"entity_ids": ["entity-1"]}]

    def _extract_entity_ids_from_facts(self, top_k_facts):  # noqa: ARG002
        return ["entity-1"]

    def _expand_subgraph(self, seed_entity_ids, entity_relevance_scores=None, owner_id=None):  # noqa: ARG002
        return {"entity-1"}, {"chunk-1", "chunk-3"}

    def _convert_to_chunks(self, chunk_ids, chunk_scores, owner_id):  # noqa: ARG002
        chunks = []
        for chunk_id, score in zip(chunk_ids, chunk_scores):
            chunks.append(
                Chunk(
                    content=f"content for {chunk_id}",
                    metadata={"score": score, "file_name": f"{chunk_id}.md"},
                    id=chunk_id,
                )
            )
        return chunks


class _StubGraphRetrieverNoSeeds:
    def __init__(self):
        self.config = types.SimpleNamespace(
            fact_retrieval_top_k=1,
            seed_entities_from_entity_nn_enabled=False,
            enable_pruning=False,
            dense_mix_in_top_k=0,
        )

    def _get_fact_scores_faiss(self, query, owner_id, query_doc_scores=None):  # noqa: ARG002
        return np.array([]), []


class _StubAdapter:
    def __init__(self, retriever):
        self.retriever = retriever
        self.supports_concurrent_calls = True

    async def acypher(self, cypher, params=None, *, access_scope=None):  # noqa: ARG002
        return []

    def cypher_capable(self):  # noqa: D401
        return True

    def metadata(self):
        return {"capabilities": [{"name": "cypher_query"}]}


def _chunk(content, chunk_id, score):
    return Chunk(content=content, metadata={"score": score, "file_name": f"{chunk_id}.md"}, id=chunk_id)


@pytest.mark.asyncio
async def test_search_tool_combines_channels() -> None:
    dense = _StubDenseRetriever([_chunk("dense one", "dense-1", 0.9)])
    bm25 = _StubBM25Retriever([_chunk("bm25 one", "bm25-1", 0.8)])
    adapter = _StubAdapter(_StubGraphRetriever())
    tool = SearchTool(llm_connector=_StubLLM(), dense_retriever=dense, bm25_retriever=bm25)
    request = ToolRunRequest(
        question="find relevant chunks",
        plan_step="plan_01",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={},
    )

    result = await tool.run(request)
    sources = {evidence.source for evidence in result.evidences}
    assert {"faiss", "bm25", "graph_chunk"}.issubset(sources)
    assert result.diagnostics.get("channels") == ["faiss", "bm25", "graph_chunk"]


@pytest.mark.asyncio
async def test_graph_chunk_raises_without_llm_when_fallback_needed() -> None:
    adapter = _StubAdapter(_StubGraphRetrieverNoSeeds())
    tool = SearchGraphChunkTool(llm_connector=None)
    request = ToolRunRequest(
        question="need entities",
        plan_step="plan_02",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={},
    )

    with pytest.raises(RuntimeError, match="LLM"):
        await tool.run(request)


@pytest.mark.asyncio
async def test_search_tool_filters_by_section_ids() -> None:
    dense = _StubDenseRetriever(
        [
            Chunk(content="dense s1", metadata={"score": 0.9, "file_name": "a.md", "section_id": "s1"}, id="d1"),
            Chunk(content="dense s2", metadata={"score": 0.8, "file_name": "a.md", "section_id": "s2"}, id="d2"),
        ]
    )
    bm25 = _StubBM25Retriever(
        [
            Chunk(content="bm25 s2", metadata={"score": 0.9, "file_name": "a.md", "section_id": "s2"}, id="b1"),
            Chunk(content="bm25 s1", metadata={"score": 0.8, "file_name": "a.md", "section_id": "s1"}, id="b2"),
        ]
    )
    tool = SearchTool(llm_connector=_StubLLM(), dense_retriever=dense, bm25_retriever=bm25)
    request = ToolRunRequest(
        question="find relevant chunks",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"channels": ["faiss", "bm25"], "section_ids": ["s1"], "top_k": 10},
    )

    result = await tool.run(request)
    assert result.evidences, "expected at least one evidence after filtering"
    for ev in result.evidences:
        meta = (ev.provenance or {}).get("metadata") or {}
        assert meta.get("section_id") == "s1"
