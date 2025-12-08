import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from application.rag_inference.module import RAGInference


def _make_rag_with_store(store):
    rag = object.__new__(RAGInference)
    rag.graph_retriever = SimpleNamespace(graph_store=store)
    rag.retriever = SimpleNamespace(retrievers=[], config=SimpleNamespace(built_retrievers=[]))
    rag.query_rewriter = SimpleNamespace()
    rag.reranker = SimpleNamespace()
    rag.llm = SimpleNamespace()
    rag.config = SimpleNamespace()
    return rag


def test_export_graph_overview_uses_neo4j_exporter():
    store_cls = type("PrunedHippoRAGNeo4jStore", (), {})  # mimic class name
    store = store_cls()
    owner_id = uuid.uuid4()
    overview_payload = {"chunks": [], "nodes": [], "edges": [], "metadata": {}}

    rag = _make_rag_with_store(store)

    with patch(
        "encapsulation.database.utils.graph_export_utils_neo4j.GraphExporterNeo4j.export_full_graph",
        return_value=overview_payload,
    ) as mock_export:
        result = rag.export_graph_overview(
            owner_id=owner_id,
            max_nodes=321,
            max_edges=654,
            include_node_types=["chunk", "entity"],
        )

    assert result is overview_payload
    assert mock_export.call_count == 1
    _, kwargs = mock_export.call_args
    assert kwargs["owner_id"] == str(owner_id)
    assert kwargs["max_nodes"] == 321
    assert kwargs["max_edges"] == 654
    assert kwargs["include_node_types"] == ["chunk", "entity"]


def test_export_graph_overview_falls_back_to_igraph_exporter():
    store_cls = type("PrunedHippoRAGIGraphStore", (), {})
    store = store_cls()
    rag = _make_rag_with_store(store)
    payload = {"chunks": [{"id": "c"}], "nodes": [], "edges": [], "metadata": {}}

    with patch(
        "encapsulation.database.utils.graph_export_utils.GraphExporter.export_full_graph",
        return_value=payload,
    ) as mock_export:
        result = rag.export_graph_overview(owner_id=None, max_nodes=100, max_edges=200, include_node_types=None)

    assert result is payload
    _, kwargs = mock_export.call_args
    assert kwargs["owner_id"] is None
    assert kwargs["max_nodes"] == 100
    assert kwargs["max_edges"] == 200


def test_export_graph_overview_raises_when_no_store():
    rag = object.__new__(RAGInference)
    rag.graph_retriever = None
    rag.retriever = SimpleNamespace(
        retrievers=[],
        config=SimpleNamespace(built_retrievers=[]),
    )
    rag.query_rewriter = SimpleNamespace()
    rag.reranker = SimpleNamespace()
    rag.llm = SimpleNamespace()
    rag.config = SimpleNamespace()

    with pytest.raises(RuntimeError):
        rag.export_graph_overview(owner_id=None)
