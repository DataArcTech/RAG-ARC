import uuid
from types import SimpleNamespace
from unittest.mock import patch

from encapsulation.data_model.schema import Chunk

from application.rag_inference.module import RAGInference


class _StubKnowledge:
    def is_file_active(self, file_id: str) -> bool:  # noqa: ARG002
        return True


class _StubRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, query: str, **kwargs):  # noqa: ANN001
        owner_id = kwargs.get("owner_id")
        return_subgraph_info = bool(kwargs.get("return_subgraph_info"))
        self.calls.append({"query": query, "owner_id": owner_id, "return_subgraph_info": return_subgraph_info, "k": kwargs.get("k")})

        chunk = Chunk(
            id=f"chunk-{owner_id}",
            content=f"content-{owner_id}",
            owner_id=str(owner_id),
            metadata={"source_file_id": f"file-{owner_id}", "filename": f"file-{owner_id}.md"},
        )
        if return_subgraph_info:
            chunk.metadata["_subgraph_info"] = {
                "subgraph_nodes": [f"node-{owner_id}"],
                "seed_entity_ids": [f"entity-{owner_id}"],
                "retrieved_chunk_ids": [chunk.id],
                "node_ppr_scores": {},
                "query": query,
            }
        return [chunk]


def _make_rag():
    rag = object.__new__(RAGInference)
    rag._knowledge_module = _StubKnowledge()
    rag.query_rewriter = SimpleNamespace(rewrite_query=lambda q: q)
    rag.reranker = SimpleNamespace(rerank=lambda _q, chunks, top_k=5: list(chunks)[:top_k], get_reranker_info=lambda: {})
    rag.config = SimpleNamespace(
        candidate_selection=SimpleNamespace(graph_candidates_k=3, web_candidates_k=0, rerank_keep_k=5),
        web_search=SimpleNamespace(enabled=False, timeout_seconds=0.1, timeout_grace_seconds=0.0),
    )
    rag.llm = SimpleNamespace()
    rag.graph_retriever = SimpleNamespace(graph_store=type("PrunedHippoRAGNeo4jStore", (), {})())
    return rag


def test_rag_inference_retrieves_only_me_by_default():
    rag = _make_rag()
    rag.retriever = _StubRetriever()

    _messages, chunks, subgraph_data, subgraph_info = rag._build_messages_and_context(
        query="hello",
        owner_id=uuid.uuid4(),
        return_subgraph=False,
        include_share=False,
    )

    assert subgraph_data is None
    assert subgraph_info is None
    assert len(chunks) == 1
    assert len(rag.retriever.calls) == 1


def test_rag_inference_retrieves_me_plus_share_when_enabled_and_exports_merged_subgraph():
    rag = _make_rag()
    rag.retriever = _StubRetriever()

    me = uuid.uuid4()
    share = uuid.uuid4()

    def fake_export_subgraph(*, subgraph_node_ids, **_kwargs):  # noqa: ANN001
        node_id = sorted(list(subgraph_node_ids))[0]
        return {
            "chunks": [{"id": f"chunk-for-{node_id}"}],
            "nodes": [{"id": node_id}],
            "edges": [{"id": f"edge-for-{node_id}", "source": node_id, "target": node_id, "relation": "self"}],
            "metadata": {},
        }

    with patch(
        "encapsulation.database.utils.graph_export_utils_neo4j.GraphExporterNeo4j.export_subgraph",
        side_effect=fake_export_subgraph,
    ) as mock_export:
        _messages, chunks, subgraph_data, subgraph_info = rag._build_messages_and_context(
            query="hello",
            owner_id=me,
            return_subgraph=True,
            include_share=True,
            share_owner_id=share,
        )

    assert len(rag.retriever.calls) == 2
    assert {call["owner_id"] for call in rag.retriever.calls} == {str(me), str(share)}
    assert len(chunks) == 2
    assert isinstance(subgraph_info, dict) and subgraph_info.get("_multi_owner") is True
    assert mock_export.call_count == 2
    assert subgraph_data is not None
    assert {node["id"] for node in subgraph_data["nodes"]} == {f"node-{me}", f"node-{share}"}

