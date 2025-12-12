"""Helpers to construct concise graph chains from DeepSearch traversal data."""
import logging
from typing import Any, Dict, List, Optional, Set

from application.rag_inference.graph_store_provider import get_graph_store
from config.output_limits import (
    DEEPSEARCH_GRAPH_NODE_LIMIT,
    DEEPSEARCH_GRAPH_EDGE_LIMIT,
)

logger = logging.getLogger(__name__)


def export_subgraph_snapshot(subgraph_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the exported subgraph (nodes/edges) for downstream rendering."""

    if not subgraph_info:
        return None
    graph_store = get_graph_store()
    if graph_store is None:
        logger.warning("Graph store unavailable; skipping graph chain generation")
        return None

    node_ids: Set[str] = set(subgraph_info.get("subgraph_nodes") or [])
    if not node_ids:
        return None

    seed_ids = set(subgraph_info.get("seed_entity_ids") or [])
    retrieved_chunks = list(subgraph_info.get("retrieved_chunk_ids") or [])
    node_scores = subgraph_info.get("node_ppr_scores") or {}

    store_name = graph_store.__class__.__name__
    try:
        if store_name == "PrunedHippoRAGNeo4jStore":
            from encapsulation.database.utils.graph_export_utils_neo4j import (
                GraphExporterNeo4j as GraphExporter,
            )

            return GraphExporter.export_subgraph(
                graph_store=graph_store,
                subgraph_node_ids=node_ids,
                seed_entity_ids=seed_ids,
                retrieved_chunk_ids=retrieved_chunks,
                node_ppr_scores=node_scores,
            )
        from encapsulation.database.utils.graph_export_utils import GraphExporter

        try:
            node_indices = {int(idx) for idx in node_ids}
        except ValueError:
            node_indices = set()
        return GraphExporter.export_subgraph(
            graph_store=graph_store,
            subgraph_node_indices=node_indices,
            seed_entity_ids=seed_ids,
            retrieved_chunk_ids=retrieved_chunks,
            node_ppr_scores=node_scores,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to export graph snapshot: %s", exc)
        return None


def build_graph_chain(subgraph_info: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> List[str]:
    """Export a compact graph chain from subgraph diagnostics."""

    if not subgraph_info:
        return []

    subgraph = snapshot or export_subgraph_snapshot(subgraph_info)
    if not subgraph:
        return []

    nodes = subgraph.get("nodes") or []
    allowed_tokens: Set[str] = set()
    if nodes and DEEPSEARCH_GRAPH_NODE_LIMIT and DEEPSEARCH_GRAPH_NODE_LIMIT > 0:
        ranked_nodes = sorted(nodes, key=lambda node: node.get("ppr_score") or 0.0, reverse=True)
        selected = ranked_nodes[:DEEPSEARCH_GRAPH_NODE_LIMIT]
    else:
        selected = nodes

    for node in selected:
        token_id = str(node.get("id") or "").strip()
        if token_id:
            allowed_tokens.add(token_id)
        node_name = str(node.get("name") or "").strip()
        if node_name:
            allowed_tokens.add(node_name)

    edges = subgraph.get("edges") or []
    if not edges:
        return []

    filtered_edges = []
    for edge in edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if allowed_tokens and (source not in allowed_tokens or target not in allowed_tokens):
            continue
        filtered_edges.append(edge)

    candidate_edges = filtered_edges or edges
    entity_edges = [edge for edge in candidate_edges if edge.get("relation") != "mentions"]
    ranked = sorted(entity_edges or candidate_edges, key=lambda item: item.get("weight") or 0, reverse=True)

    if DEEPSEARCH_GRAPH_EDGE_LIMIT and DEEPSEARCH_GRAPH_EDGE_LIMIT > 0:
        ranked = ranked[:DEEPSEARCH_GRAPH_EDGE_LIMIT]

    chain: List[str] = []
    for edge in ranked:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        relation = edge.get("relation") or "related"
        if not source or not target:
            continue
        chain.append(f"{source} -[{relation}]-> {target}")
    return chain
