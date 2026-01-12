import uuid
from types import SimpleNamespace
from unittest.mock import patch

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


def test_export_graph_overview_multi_owner_calls_exporter_for_each_owner_and_merges():
    store_cls = type("PrunedHippoRAGNeo4jStore", (), {})  # mimic class name dispatch
    store = store_cls()
    rag = _make_rag_with_store(store)

    me = uuid.uuid4()
    share = uuid.uuid4()

    def fake_export_full_graph(*, owner_id, owner_scope_label, **_kwargs):  # noqa: ANN001
        return {
            "chunks": [],
            "nodes": [{"id": f"node-{owner_id}"}],
            "edges": [{"id": f"edge-{owner_id}", "source": f"node-{owner_id}", "target": f"node-{owner_id}", "relation": "self"}],
            "metadata": {"owner_scope_label": owner_scope_label},
        }

    with patch(
        "encapsulation.database.utils.graph_export_utils_neo4j.GraphExporterNeo4j.export_full_graph",
        side_effect=fake_export_full_graph,
    ) as mock_export:
        payload = rag.export_graph_overview(
            owner_id=me,
            max_nodes=10,
            max_edges=10,
            include_node_types=["chunk", "entity"],
            include_share=True,
            share_owner_id=share,
        )

    assert mock_export.call_count == 2
    called_owner_ids = {kwargs["owner_id"] for _args, kwargs in mock_export.call_args_list}
    assert called_owner_ids == {str(me), str(share)}
    assert {n["id"] for n in payload["nodes"]} == {f"node-{me}", f"node-{share}"}

