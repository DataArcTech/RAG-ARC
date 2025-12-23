from typing import Any, Optional

from fastapi import HTTPException, status


def resolve_graph_store(rag_inference: Any) -> Any | None:
    graph_store = None
    if hasattr(rag_inference, "retriever") and hasattr(rag_inference.retriever, "graph_store"):
        graph_store = rag_inference.retriever.graph_store
    elif hasattr(rag_inference, "retriever") and hasattr(rag_inference.retriever, "retrievers"):
        for sub_retriever in rag_inference.retriever.retrievers:
            if hasattr(sub_retriever, "graph_store"):
                graph_store = sub_retriever.graph_store
                break
    return graph_store


def resolve_graph_exporter(graph_store: Any):
    graph_store_class_name = graph_store.__class__.__name__
    if graph_store_class_name == "PrunedHippoRAGNeo4jStore":
        from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j as GraphExporter
    else:
        from encapsulation.database.utils.graph_export_utils import GraphExporter
    return GraphExporter


def export_full_graph_payload(
    *,
    rag_inference: Any,
    owner_scope: str,
    max_nodes: int = 500,
    max_edges: int = 2000,
    include_node_types: Optional[list[str]] = None,
) -> dict[str, Any]:
    graph_store = resolve_graph_store(rag_inference)
    if graph_store is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current retriever does not support graph visualization",
        )
    exporter = resolve_graph_exporter(graph_store)
    return exporter.export_full_graph(
        graph_store=graph_store,
        max_nodes=max_nodes,
        max_edges=max_edges,
        include_node_types=include_node_types,
        owner_id=owner_scope,
        owner_scope_label=owner_scope,
    )

