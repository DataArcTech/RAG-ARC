"""
Graph Export Utilities for Visualization

Provides utilities to export graph data from PrunedHippoRAGIGraphStore
for frontend visualization using Cytoscape.js or other graph libraries.
"""

import logging
import re
from typing import Dict, List, Any, Set, Optional

from config.output_limits import (
    GRAPH_EXPORT_CHUNK_CONTENT_PREVIEW_CHARS,
    GRAPH_EXPORT_EDGE_FETCH_FACTOR,
    GRAPH_EXPORT_EDGE_FETCH_MAX,
)

logger = logging.getLogger(__name__)


def _truncate_text(text: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "…"


class GraphExporter:
    """Export graph data for visualization"""

    @staticmethod
    def _should_filter_entity(entity_name: str) -> bool:
        """
        Check if an entity should be filtered out

        Filters out:
        - Pure numbers (integers, decimals)
        - Timestamps (Unix timestamps, millisecond timestamps)
        - Common time formats (YYYY-MM-DD, HH:MM:SS, etc.)

        Args:
            entity_name: Entity name to check

        Returns:
            True if entity should be filtered out, False otherwise
        """
        if not entity_name or not isinstance(entity_name, str):
            return False

        entity_name = entity_name.strip()

        # Check for pure numbers (integers or decimals)
        if re.match(r'^-?\d+(\.\d+)?$', entity_name):
            return True

        # Check for Unix timestamps (10 digits) or millisecond timestamps (13 digits)
        if re.match(r'^\d{10}$', entity_name) or re.match(r'^\d{13}$', entity_name):
            return True

        # Check for common date/time formats
        # YYYY-MM-DD, YYYY/MM/DD, YYYY-MM-DD HH:MM:SS, etc.
        if re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', entity_name):
            return True

        # Check for time format HH:MM:SS or HH:MM
        if re.match(r'^\d{2}:\d{2}(:\d{2})?$', entity_name):
            return True

        return False

    @staticmethod
    def export_full_graph(
        graph_store,
        max_nodes: int = 1000,
        max_edges: int = 5000,
        include_node_types: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
        owner_scope_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export complete graph for visualization

        Args:
            graph_store: PrunedHippoRAGIGraphStore instance
            max_nodes: Maximum number of nodes to export (for performance)
            max_edges: Maximum number of edges to export
            include_node_types: List of node types to include ['chunk', 'entity', 'fact']
                               If None, include all types
            owner_id: Optional owner scope; when provided, only export data for that owner

        Returns:
            Dict with 'chunks', 'nodes' (entities), 'edges', and 'metadata'
        """
        if include_node_types is None:
            include_node_types = ['chunk', 'entity', 'fact']

        # Normalize owner scope using graph store helper when available
        owner_filter = None
        normalizer = getattr(graph_store, "_normalize_owner_id", None)
        if owner_id is not None:
            owner_filter = normalizer(owner_id) if normalizer else str(owner_id)

        chunks = []
        nodes = []  # Only entities
        edges = []
        categories_set = set()  # Track unique entity types

        graph = graph_store.graph
        idx_to_node = graph_store.idx_to_node

        # Get node statistics
        total_nodes = graph.vcount()
        logger.info(f"Exporting graph with {total_nodes} nodes")

        # Sample nodes if too many
        if total_nodes > max_nodes:
            logger.warning(f"Graph has {total_nodes} nodes, sampling {max_nodes} nodes")
            # Sample nodes by degree (keep high-degree nodes)
            degrees = graph.degree()
            node_indices = sorted(range(total_nodes), key=lambda i: degrees[i], reverse=True)[:max_nodes]
        else:
            node_indices = range(total_nodes)

        cursor = graph_store.conn.cursor()

        # Load only metadata needed for the sampled node set (avoid scanning full tables).
        candidate_chunk_ids: List[str] = []
        candidate_entity_ids: List[str] = []
        for idx in node_indices:
            node_id = idx_to_node.get(idx)
            if not node_id:
                continue
            node_type = graph.vs[idx]["node_type"]
            if node_type == "chunk":
                candidate_chunk_ids.append(node_id)
            elif node_type == "entity":
                candidate_entity_ids.append(node_id)

        chunk_info: Dict[str, Dict[str, Any]] = {}
        if candidate_chunk_ids:
            placeholders = ",".join(["?"] * len(candidate_chunk_ids))
            chunk_query = f"SELECT chunk_id, content, owner_id FROM chunks WHERE chunk_id IN ({placeholders})"
            params: List[Any] = list(candidate_chunk_ids)
            if owner_filter is not None:
                chunk_query += " AND owner_id = ?"
                params.append(owner_filter)
            cursor.execute(chunk_query, tuple(params))
            for chunk_id, content, owner in cursor.fetchall():
                chunk_info[str(chunk_id)] = {
                    "content": _truncate_text(content or "", limit=int(GRAPH_EXPORT_CHUNK_CONTENT_PREVIEW_CHARS)),
                    "owner_id": (owner or None),
                }

        entity_id_to_info: Dict[str, tuple[Any, Any, Any]] = {}
        if candidate_entity_ids:
            placeholders = ",".join(["?"] * len(candidate_entity_ids))
            entity_query = (
                f"SELECT entity_id, entity_name, entity_type, owner_id FROM entities WHERE entity_id IN ({placeholders})"
            )
            params = list(candidate_entity_ids)
            if owner_filter is not None:
                entity_query += " AND owner_id = ?"
                params.append(owner_filter)
            cursor.execute(entity_query, tuple(params))
            entity_id_to_info = {eid: (name, etype, owner or None) for eid, name, etype, owner in cursor.fetchall()}

        # Export nodes - separate chunks and entities
        node_set = set()
        selected_entity_names: List[str] = []
        for idx in node_indices:
            node_id = idx_to_node.get(idx)
            if not node_id:
                continue

            # Determine node type
            node_type = graph.vs[idx]['node_type']

            # Filter by node type
            if node_type not in include_node_types:
                continue

            # Build node structure based on type
            if node_type == 'chunk':
                chunk_meta = chunk_info.get(node_id)
                if chunk_meta is None:
                    continue
                node_set.add(idx)
                chunk_obj = {'id': node_id, 'type': 'chunk'}
                if chunk_meta.get('content'):
                    chunk_obj['content'] = chunk_meta['content']
                chunks.append(chunk_obj)

            elif node_type == 'entity':
                entity_info = entity_id_to_info.get(node_id)
                if not entity_info:
                    continue
                entity_name, entity_type, _ = entity_info

                # Filter out entities that are pure numbers, timestamps, or time formats
                if GraphExporter._should_filter_entity(entity_name):
                    continue

                node_set.add(idx)
                entity_obj = {
                    "id": node_id,
                    "type": "entity",
                    "name": entity_name,
                    "category": entity_type or "Entity",
                }
                categories_set.add(entity_type or "Entity")
                nodes.append(entity_obj)
                selected_entity_names.append(entity_name)

        # Build (head, tail) -> relation map for O(1) lookups during edge rendering.
        fact_pair_to_relation: Dict[tuple[str, str], str] = {}
        if selected_entity_names:
            placeholders = ",".join(["?"] * len(selected_entity_names))
            fact_query = f"SELECT head, relation, tail, owner_id FROM facts WHERE head IN ({placeholders}) OR tail IN ({placeholders})"
            params = list(selected_entity_names) + list(selected_entity_names)
            if owner_filter is not None:
                fact_query += " AND owner_id = ?"
                params.append(owner_filter)
            cursor.execute(fact_query, tuple(params))
            for head, relation, tail, owner in cursor.fetchall():
                if owner_filter is not None and owner != owner_filter:
                    continue
                head_text = str(head or "")
                tail_text = str(tail or "")
                rel_text = str(relation or "").strip()
                if not head_text or not tail_text or not rel_text:
                    continue
                fact_pair_to_relation.setdefault((head_text, tail_text), rel_text)

        # Collect edges by type for uniform sampling
        edges_by_type = {
            'mentions': [],
            'fact_relation': [],
            'synonymy': [],
            'other': []
        }

        max_edges = max(0, int(max_edges or 0))
        if max_edges <= 0:
            edges = []
        else:
            node_idx_list = sorted(node_set)
            subgraph = graph.induced_subgraph(node_idx_list)
            index_to_node_id = [idx_to_node.get(idx) for idx in node_idx_list]
            index_to_node_type = [subgraph.vs[i]["node_type"] for i in range(subgraph.vcount())]

            factor = max(1, int(GRAPH_EXPORT_EDGE_FETCH_FACTOR))
            fetch_cap = max_edges * factor
            fetch_cap = min(int(GRAPH_EXPORT_EDGE_FETCH_MAX), fetch_cap) if fetch_cap > 0 else 0
            if fetch_cap <= 0:
                fetch_cap = min(int(GRAPH_EXPORT_EDGE_FETCH_MAX), max_edges)

            for edge in subgraph.es[:fetch_cap]:
                source_idx = int(edge.source)
                target_idx = int(edge.target)

                source_id = index_to_node_id[source_idx]
                target_id = index_to_node_id[target_idx]

                if not source_id or not target_id:
                    continue

                weight = edge["weight"] if "weight" in edge.attributes() else 1.0

                # Determine edge type and relation
                source_type = index_to_node_type[source_idx]
                target_type = index_to_node_type[target_idx]

                # Build edge object with source/target as entity_name or chunk_id
                edge_obj = {
                    "id": f"{source_id}_{target_id}",
                    "weight": weight,
                }

                # Determine relation and source/target display
                if source_type == "chunk" and target_type == "entity":
                    edge_obj["source"] = source_id
                    entity_info = entity_id_to_info.get(target_id)
                    edge_obj["target"] = entity_info[0] if entity_info else target_id
                    edge_obj["relation"] = "mentions"
                    edges_by_type["mentions"].append(edge_obj)
                elif source_type == "entity" and target_type == "chunk":
                    continue
                elif source_type == "entity" and target_type == "entity":
                    source_info = entity_id_to_info.get(source_id)
                    target_info = entity_id_to_info.get(target_id)
                    source_name = source_info[0] if source_info else source_id
                    target_name = target_info[0] if target_info else target_id

                    relation = fact_pair_to_relation.get((str(source_name), str(target_name)))
                    if relation:
                        edge_obj["source"] = source_name
                        edge_obj["target"] = target_name
                        edge_obj["relation"] = relation
                        edges_by_type["fact_relation"].append(edge_obj)
                        continue
                    relation = fact_pair_to_relation.get((str(target_name), str(source_name)))
                    if relation:
                        edge_obj["source"] = target_name
                        edge_obj["target"] = source_name
                        edge_obj["relation"] = relation
                        edges_by_type["fact_relation"].append(edge_obj)
                else:
                    edge_obj["source"] = source_id
                    edge_obj["target"] = target_id
                    edge_obj["relation"] = "related"
                    edges_by_type["other"].append(edge_obj)

        # Uniformly sample edges from different types
        # Allocate edge quota: mentions (50%), fact_relation (35%), synonymy (10%), other (5%)
        mentions_quota = int(max_edges * 0.50)
        fact_quota = int(max_edges * 0.35)
        synonymy_quota = int(max_edges * 0.10)
        other_quota = max_edges - mentions_quota - fact_quota - synonymy_quota

        # Sample edges based on quota
        edges.extend(edges_by_type['mentions'][:mentions_quota])
        edges.extend(edges_by_type['fact_relation'][:fact_quota])
        edges.extend(edges_by_type['synonymy'][:synonymy_quota])
        edges.extend(edges_by_type['other'][:other_quota])

        logger.info(f"Exported {len(chunks)} chunks, {len(nodes)} entities, and {len(edges)} edges")

        # Build categories list from unique entity types
        categories = [{'name': cat} for cat in sorted(categories_set)]

        scope_value = owner_scope_label if owner_scope_label is not None else owner_filter

        return {
            'chunks': chunks,
            'nodes': nodes,  # Only entities
            'edges': edges,
            'metadata': {
                'total_nodes': total_nodes,
                'total_edges': graph.ecount(),
                'exported_nodes': len(chunks) + len(nodes),
                'exported_edges': len(edges),
                'sampled': total_nodes > max_nodes,
                'categories': categories,
                'owner_scope': scope_value,
            }
        }
    
    @staticmethod
    def export_subgraph(
        graph_store,
        subgraph_node_indices: Set[int],
        seed_entity_ids: Optional[Set[str]] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        node_ppr_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Export retrieval subgraph for visualization

        Args:
            graph_store: PrunedHippoRAGIGraphStore instance
            subgraph_node_indices: Set of node indices in the subgraph
            seed_entity_ids: Set of seed entity IDs (highlighted)
            retrieved_chunk_ids: List of retrieved chunk IDs (highlighted and ordered)
            node_ppr_scores: Dict mapping node_id to PPR score

        Returns:
            Dict with 'chunks', 'nodes' (entities), 'edges', and 'metadata'
        """
        chunks = []
        nodes = []  # Only entities
        edges = []
        categories_set = set()  # Track unique entity types

        graph = graph_store.graph
        idx_to_node = graph_store.idx_to_node

        seed_entity_ids = seed_entity_ids or set()
        retrieved_chunk_ids = retrieved_chunk_ids or []
        node_ppr_scores = node_ppr_scores or {}

        # Build entity_id to entity_name and entity_type mapping
        cursor = graph_store.conn.cursor()
        cursor.execute('SELECT entity_id, entity_name, entity_type FROM entities')
        entity_id_to_info = {eid: (name, etype) for eid, name, etype in cursor.fetchall()}

        # Export nodes - separate chunks and entities
        for idx in subgraph_node_indices:
            node_id = idx_to_node.get(idx)
            if not node_id:
                continue

            node_type = graph.vs[idx]['node_type']

            # Build node structure based on type
            if node_type == 'chunk':
                cursor.execute('SELECT content FROM chunks WHERE chunk_id = ?', (node_id,))
                row = cursor.fetchone()
                chunk_obj = {
                    'id': node_id,
                    'type': 'chunk'
                }
                if row:
                    chunk_obj['content'] = row[0]
                # Add PPR score if available
                if node_id in node_ppr_scores:
                    chunk_obj['ppr_score'] = node_ppr_scores[node_id]
                chunks.append(chunk_obj)

            elif node_type == 'entity':
                entity_info = entity_id_to_info.get(node_id)
                entity_obj = {
                    'id': node_id,
                    'type': 'entity'
                }
                if entity_info:
                    entity_name, entity_type = entity_info

                    # Filter out entities that are pure numbers, timestamps, or time formats
                    if GraphExporter._should_filter_entity(entity_name):
                        continue

                    entity_obj = {
                        'id': node_id
                    }
                    # Use 'name' instead of 'entity_name'
                    entity_obj['name'] = entity_name
                    # Use 'category' for entity_type
                    entity_obj['category'] = entity_type or 'Entity'
                    categories_set.add(entity_type or 'Entity')
                    
                    # Add weight field (default to 2 for entity nodes, 1 for seed entities)
                    if node_id in seed_entity_ids:
                        entity_obj['weight'] = 1
                        entity_obj['is_seed'] = True
                    else:
                        entity_obj['weight'] = 2

                    # Add PPR score if available
                    if node_id in node_ppr_scores:
                        entity_obj['ppr_score'] = node_ppr_scores[node_id]

                    nodes.append(entity_obj)
        
        # Build fact_id to relation mapping
        cursor.execute('SELECT fact_id, head, relation, tail FROM facts')
        fact_relations = {fid: (head, relation, tail) for fid, head, relation, tail in cursor.fetchall()}

        # Export edges (no limit for subgraph, but still skip reverse edges)
        subgraph_node_set = set(subgraph_node_indices)
        for edge in graph.es:
            source_idx = edge.source
            target_idx = edge.target

            # Only include edges within subgraph
            if source_idx not in subgraph_node_set or target_idx not in subgraph_node_set:
                continue

            source_id = idx_to_node.get(source_idx)
            target_id = idx_to_node.get(target_idx)

            if not source_id or not target_id:
                continue

            weight = edge['weight'] if 'weight' in edge.attributes() else 1.0

            # Determine edge type and relation
            source_type = graph.vs[source_idx]['node_type']
            target_type = graph.vs[target_idx]['node_type']

            # Build edge object with source/target as entity_name or chunk_id
            edge_obj = {
                'id': f"{source_id}_{target_id}",
                'weight': weight
            }

            # Determine relation and source/target display
            if source_type == 'chunk' and target_type == 'entity':
                # Chunk mentions entity
                edge_obj['source'] = source_id  # chunk_id
                # Get entity name from entity_id_to_info
                entity_info = entity_id_to_info.get(target_id)
                edge_obj['target'] = entity_info[0] if entity_info else target_id  # entity_name
                edge_obj['relation'] = 'mentions'

            elif source_type == 'entity' and target_type == 'chunk':
                # Skip reverse edge (mentioned_by), as edges are undirected
                continue

            elif source_type == 'entity' and target_type == 'entity':
                # Entity-entity relation (synonymy or fact-based)
                source_info = entity_id_to_info.get(source_id)
                target_info = entity_id_to_info.get(target_id)
                source_name = source_info[0] if source_info else source_id
                target_name = target_info[0] if target_info else target_id

                # Try to find relation from facts and determine correct direction
                relation_found = False
                for _, (head, relation, tail) in fact_relations.items():
                    if head == source_name and tail == target_name:
                        # Direction matches: source -> target
                        edge_obj['source'] = source_name
                        edge_obj['target'] = target_name
                        edge_obj['relation'] = relation
                        relation_found = True
                        break
                    elif head == target_name and tail == source_name:
                        # Direction reversed: need to swap source and target
                        edge_obj['source'] = target_name
                        edge_obj['target'] = source_name
                        edge_obj['relation'] = relation
                        relation_found = True
                        break

                # If no fact relation found, skip (no synonymy edges anymore)
                # SIMILAR_TO relationships are filtered out
                if not relation_found:
                    continue

            else:
                # Fallback for other edge types
                edge_obj['source'] = source_id
                edge_obj['target'] = target_id
                edge_obj['relation'] = 'related'

            edges.append(edge_obj)

        logger.info(f"Exported subgraph: {len(chunks)} chunks, {len(nodes)} entities, {len(edges)} edges")

        # Build categories list from unique entity types
        categories = [{'name': cat} for cat in sorted(categories_set)]

        return {
            'chunks': chunks,
            'nodes': nodes,  # Only entities
            'edges': edges,
            'metadata': {
                'total_nodes': len(chunks) + len(nodes),
                'total_edges': len(edges),
                'seed_entities': len(seed_entity_ids),
                'retrieved_chunks': len(retrieved_chunk_ids),
                'categories': categories
            }
        }
