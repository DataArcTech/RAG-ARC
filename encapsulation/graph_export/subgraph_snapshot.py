"""Graph-store specific subgraph export helpers (infrastructure layer)."""
import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


def export_subgraph_snapshot(
    subgraph_info: Dict[str, Any],
    *,
    graph_store: Any | None = None,
) -> Optional[Dict[str, Any]]:
    """Return an exported subgraph snapshot (nodes/edges/chunks) when supported by the graph_store."""

    if not subgraph_info:
        return None
    if graph_store is None:
        logger.warning("Graph store unavailable; skipping graph snapshot export")
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

