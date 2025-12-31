from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from core.retrieval.graph_retrieveal.neo4j_queries import get_neo4j_query


class GraphQueryExecutor(Protocol):
    def execute(self, query_type: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass(frozen=True)
class Neo4jGraphQueryExecutor:
    graph_store: Any

    def execute(self, query_type: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        query = get_neo4j_query(query_type)
        return self.graph_store.query(query, params or {})


@dataclass(frozen=True)
class NetworkXGraphQueryExecutor:
    graph_store: Any

    def execute(self, query_type: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        params = params or {}
        results: List[Dict[str, Any]] = []
        graph = self.graph_store.graph

        if query_type == "semantic_search_entities":
            for node_id, node_data in graph.nodes(data=True):
                if node_data.get("node_type") != "Entity" or not node_data.get("embedding"):
                    continue
                mention_count = 0
                for _, target, edge_data in graph.edges(data=True):
                    if target == node_id and edge_data.get("relation_type") == "MENTIONS":
                        mention_count += 1
                results.append(
                    {
                        "entity_id": node_data.get("id_"),
                        "entity_name": node_data.get("entity_name"),
                        "entity_type": node_data.get("entity_type"),
                        "embedding": node_data.get("embedding"),
                        "mention_count": mention_count,
                        "attributes": node_data.get("attributes", "{}"),
                    }
                )
            results.sort(key=lambda x: int(x.get("mention_count") or 0), reverse=True)
            return results[:2000]

        if query_type == "semantic_search_chunks":
            for node_id, node_data in graph.nodes(data=True):
                if node_data.get("node_type") != "Chunk" or not node_data.get("embedding"):
                    continue
                entity_count = 0
                entity_names: List[str] = []
                for source, target, edge_data in graph.edges(data=True):
                    if source != node_id or edge_data.get("relation_type") != "MENTIONS":
                        continue
                    entity_count += 1
                    target_data = graph.nodes[target]
                    name = target_data.get("entity_name")
                    if name:
                        entity_names.append(str(name))
                results.append(
                    {
                        "chunk_id": node_data.get("id_"),
                        "content": node_data.get("content"),
                        "embedding": node_data.get("embedding"),
                        "metadata": node_data.get("metadata", "{}"),
                        "entity_count": entity_count,
                        "entity_names": entity_names,
                    }
                )
            results.sort(key=lambda x: int(x.get("entity_count") or 0), reverse=True)
            return results[:2000]

        if query_type == "get_entity_data":
            entity_id = params.get("entity_id")
            entity_node_id = f"entity_{entity_id}"
            if graph.has_node(entity_node_id):
                node_data = graph.nodes[entity_node_id]
                return [
                    {
                        "name": node_data.get("entity_name"),
                        "type": node_data.get("entity_type"),
                        "attributes": node_data.get("attributes", "{}"),
                    }
                ]
            return []

        if query_type == "get_entity_neighbors":
            entity_id = params.get("entity_id")
            entity_node_id = f"entity_{entity_id}"
            if not graph.has_node(entity_node_id):
                return []

            neighbors: List[Dict[str, Any]] = []

            for target in graph.successors(entity_node_id):
                edge_data = graph.get_edge_data(entity_node_id, target) or {}
                for _, edge_attrs in edge_data.items():
                    if edge_attrs.get("relation_type") == "MENTIONS":
                        continue
                    mention_count = 0
                    for _, mentioned_target, mentioned_edge_data in graph.edges(data=True):
                        if mentioned_target == target and mentioned_edge_data.get("relation_type") == "MENTIONS":
                            mention_count += 1
                    target_data = graph.nodes[target]
                    neighbors.append(
                        {
                            "neighbor_id": target_data.get("id_"),
                            "relation_type": edge_attrs.get("relation_type"),
                            "mention_count": mention_count,
                        }
                    )

            for source in graph.predecessors(entity_node_id):
                edge_data = graph.get_edge_data(source, entity_node_id) or {}
                for _, edge_attrs in edge_data.items():
                    if edge_attrs.get("relation_type") == "MENTIONS":
                        continue
                    mention_count = 0
                    for _, mentioned_target, mentioned_edge_data in graph.edges(data=True):
                        if mentioned_target == source and mentioned_edge_data.get("relation_type") == "MENTIONS":
                            mention_count += 1
                    source_data = graph.nodes[source]
                    neighbors.append(
                        {
                            "neighbor_id": source_data.get("id_"),
                            "relation_type": edge_attrs.get("relation_type"),
                            "mention_count": mention_count,
                        }
                    )

            return neighbors

        if query_type == "get_entity_degree":
            entity_id = params.get("entity_id")
            entity_node_id = f"entity_{entity_id}"
            if graph.has_node(entity_node_id):
                return [{"degree": graph.degree(entity_node_id)}]
            return [{"degree": 0}]

        if query_type == "get_entity_mentions":
            entity_id = params.get("entity_id")
            entity_node_id = f"entity_{entity_id}"
            mention_count = 0
            for _, target, edge_data in graph.edges(data=True):
                if target == entity_node_id and edge_data.get("relation_type") == "MENTIONS":
                    mention_count += 1
            return [{"mention_count": mention_count}]

        if query_type == "get_chunk_entities":
            chunk_id = params.get("chunk_id")
            chunk_node_id = f"chunk_{chunk_id}"
            entity_ids: List[Dict[str, Any]] = []
            for target in graph.successors(chunk_node_id):
                edge_data = graph.get_edge_data(chunk_node_id, target) or {}
                for _, edge_attrs in edge_data.items():
                    if edge_attrs.get("relation_type") != "MENTIONS":
                        continue
                    target_data = graph.nodes[target]
                    entity_ids.append({"entity_id": target_data.get("id_")})
            return entity_ids

        if query_type == "backtrack_chunks":
            entity_id = params.get("entity_id")
            entity_node_id = f"entity_{entity_id}"
            chunks_per_entity = int(params.get("chunks_per_entity", 10) or 10)
            chunks: List[Dict[str, Any]] = []
            for source in graph.predecessors(entity_node_id):
                edge_data = graph.get_edge_data(source, entity_node_id) or {}
                for _, edge_attrs in edge_data.items():
                    if edge_attrs.get("relation_type") != "MENTIONS":
                        continue
                    source_data = graph.nodes[source]
                    if source_data.get("node_type") != "Chunk" or not source_data.get("embedding"):
                        continue
                    chunks.append(
                        {
                            "chunk_id": source_data.get("id_"),
                            "content": source_data.get("content"),
                            "embedding": source_data.get("embedding"),
                        }
                    )
                    if len(chunks) >= chunks_per_entity:
                        break
                if len(chunks) >= chunks_per_entity:
                    break
            return chunks

        raise ValueError(f"Unknown query type: {query_type}")


def build_graph_query_executor(*, graph_store: Any, graph_store_type: str) -> GraphQueryExecutor:
    normalized = (graph_store_type or "").strip().lower()
    if normalized.startswith("neo4j"):
        return Neo4jGraphQueryExecutor(graph_store=graph_store)
    if normalized.startswith("networkx"):
        return NetworkXGraphQueryExecutor(graph_store=graph_store)
    raise ValueError(f"Unsupported graph store type: {graph_store_type}")


__all__ = ["GraphQueryExecutor", "build_graph_query_executor"]

