"""Presentation helpers to build concise graph chains from exported subgraphs."""
from typing import Any, Dict, List, Optional, Set

from config.output_limits import (
    DEEPSEARCH_GRAPH_EDGE_LIMIT,
    DEEPSEARCH_GRAPH_NODE_LIMIT,
)


def build_graph_chain(snapshot: Optional[Dict[str, Any]]) -> List[str]:
    """Build a compact edge chain from an exported subgraph snapshot."""

    if not snapshot:
        return []

    nodes = snapshot.get("nodes") or []
    allowed_tokens: Set[str] = set()
    if nodes and DEEPSEARCH_GRAPH_NODE_LIMIT and DEEPSEARCH_GRAPH_NODE_LIMIT > 0:
        ranked_nodes = sorted(nodes, key=lambda node: node.get("ppr_score") or 0.0, reverse=True)
        selected = ranked_nodes[:DEEPSEARCH_GRAPH_NODE_LIMIT]
    else:
        selected = nodes

    for node in selected:
        if not isinstance(node, dict):
            continue
        token_id = str(node.get("id") or "").strip()
        if token_id:
            allowed_tokens.add(token_id)
        node_name = str(node.get("name") or "").strip()
        if node_name:
            allowed_tokens.add(node_name)

    edges = snapshot.get("edges") or []
    if not edges:
        return []

    filtered_edges: List[Dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if allowed_tokens and (source not in allowed_tokens or target not in allowed_tokens):
            continue
        filtered_edges.append(edge)

    candidate_edges = filtered_edges or [edge for edge in edges if isinstance(edge, dict)]
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

