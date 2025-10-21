"""
Graph Export Utilities for Visualization

Provides utilities to export graph data from PrunedHippoRAGIGraphStore
for frontend visualization using Cytoscape.js or other graph libraries.
"""

import logging
from typing import Dict, List, Any, Set, Optional

logger = logging.getLogger(__name__)


class GraphExporter:
    """Export graph data for visualization"""
    
    @staticmethod
    def export_full_graph(
        graph_store,
        max_nodes: int = 1000,
        max_edges: int = 5000,
        include_node_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export complete graph for visualization
        
        Args:
            graph_store: PrunedHippoRAGIGraphStore instance
            max_nodes: Maximum number of nodes to export (for performance)
            max_edges: Maximum number of edges to export
            include_node_types: List of node types to include ['chunk', 'entity', 'fact']
                               If None, include all types
        
        Returns:
            Dict with 'nodes' and 'edges' in Cytoscape.js format
        """
        if include_node_types is None:
            include_node_types = ['chunk', 'entity', 'fact']
        
        nodes = []
        edges = []
        
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
        
        # Build entity_id to entity_name mapping
        cursor = graph_store.conn.cursor()
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entity_id_to_name = {eid: name for eid, name in cursor.fetchall()}

        # Export nodes
        node_set = set(node_indices)
        for idx in node_indices:
            node_id = idx_to_node.get(idx)
            if not node_id:
                continue

            # Determine node type
            node_type = graph.vs[idx]['node_type']

            # Filter by node type
            if node_type not in include_node_types:
                continue

            # Build simplified node structure
            node_obj = {
                'id': node_id,
                'type': node_type
            }

            # Add type-specific fields
            if node_type == 'chunk':
                cursor.execute('SELECT content FROM chunks WHERE chunk_id = ?', (node_id,))
                row = cursor.fetchone()
                if row:
                    node_obj['content'] = row[0]

            elif node_type == 'entity':
                entity_name = entity_id_to_name.get(node_id, '')
                node_obj['entity_name'] = entity_name

            nodes.append(node_obj)
        
        # Build fact_id to relation mapping
        cursor.execute('SELECT fact_id, head, relation, tail FROM facts')
        fact_relations = {fid: (head, relation, tail) for fid, head, relation, tail in cursor.fetchall()}

        # Export edges
        edge_count = 0
        for edge in graph.es:
            if edge_count >= max_edges:
                break

            source_idx = edge.source
            target_idx = edge.target

            # Only include edges between sampled nodes
            if source_idx not in node_set or target_idx not in node_set:
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
                edge_obj['source'] = source_id  # chunk_id
                edge_obj['target'] = entity_id_to_name.get(target_id, target_id)  # entity_name
                edge_obj['relation'] = 'mentions'

            elif source_type == 'entity' and target_type == 'chunk':
                edge_obj['source'] = entity_id_to_name.get(source_id, source_id)  # entity_name
                edge_obj['target'] = target_id  # chunk_id
                edge_obj['relation'] = 'mentioned_by'

            elif source_type == 'entity' and target_type == 'entity':
                source_name = entity_id_to_name.get(source_id, source_id)
                target_name = entity_id_to_name.get(target_id, target_id)

                # Try to find relation from facts and determine correct direction
                relation_found = False
                for _, (head, relation, tail) in fact_relations.items():
                    if head == source_name and tail == target_name:
                        edge_obj['source'] = source_name
                        edge_obj['target'] = target_name
                        edge_obj['relation'] = relation
                        relation_found = True
                        break
                    elif head == target_name and tail == source_name:
                        edge_obj['source'] = target_name
                        edge_obj['target'] = source_name
                        edge_obj['relation'] = relation
                        relation_found = True
                        break

                if not relation_found:
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = 'synonymy'

            else:
                edge_obj['source'] = source_id
                edge_obj['target'] = target_id
                edge_obj['relation'] = 'related'

            edges.append(edge_obj)
            edge_count += 1
        
        logger.info(f"Exported {len(nodes)} nodes and {len(edges)} edges")
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': total_nodes,
                'total_edges': graph.ecount(),
                'exported_nodes': len(nodes),
                'exported_edges': len(edges),
                'sampled': total_nodes > max_nodes
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
            Dict with 'nodes' and 'edges' in simplified format
        """
        nodes = []
        edges = []

        graph = graph_store.graph
        idx_to_node = graph_store.idx_to_node

        seed_entity_ids = seed_entity_ids or set()
        retrieved_chunk_ids = retrieved_chunk_ids or []
        node_ppr_scores = node_ppr_scores or {}

        # Build entity_id to entity_name mapping
        cursor = graph_store.conn.cursor()
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entity_id_to_name = {eid: name for eid, name in cursor.fetchall()}

        # Export nodes
        for idx in subgraph_node_indices:
            node_id = idx_to_node.get(idx)
            if not node_id:
                continue

            node_type = graph.vs[idx]['node_type']

            # Build simplified node structure
            node_obj = {
                'id': node_id,
                'type': node_type
            }

            # Add type-specific fields
            if node_type == 'chunk':
                cursor.execute('SELECT content FROM chunks WHERE chunk_id = ?', (node_id,))
                row = cursor.fetchone()
                if row:
                    node_obj['content'] = row[0]

            elif node_type == 'entity':
                entity_name = entity_id_to_name.get(node_id, '')
                node_obj['entity_name'] = entity_name

                # Mark seed entities
                if node_id in seed_entity_ids:
                    node_obj['is_seed'] = True

            # Add PPR score if available
            if node_id in node_ppr_scores:
                node_obj['ppr_score'] = node_ppr_scores[node_id]

            nodes.append(node_obj)
        
        # Build fact_id to relation mapping
        cursor.execute('SELECT fact_id, head, relation, tail FROM facts')
        fact_relations = {fid: (head, relation, tail) for fid, head, relation, tail in cursor.fetchall()}

        # Export edges
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
                edge_obj['target'] = entity_id_to_name.get(target_id, target_id)  # entity_name
                edge_obj['relation'] = 'mentions'

            elif source_type == 'entity' and target_type == 'chunk':
                # Entity mentioned by chunk
                edge_obj['source'] = entity_id_to_name.get(source_id, source_id)  # entity_name
                edge_obj['target'] = target_id  # chunk_id
                edge_obj['relation'] = 'mentioned_by'

            elif source_type == 'entity' and target_type == 'entity':
                # Entity-entity relation (synonymy or fact-based)
                source_name = entity_id_to_name.get(source_id, source_id)
                target_name = entity_id_to_name.get(target_id, target_id)

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

                if not relation_found:
                    # No fact found, assume synonymy (bidirectional)
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = 'synonymy'

            else:
                # Fallback for other edge types
                edge_obj['source'] = source_id
                edge_obj['target'] = target_id
                edge_obj['relation'] = 'related'

            edges.append(edge_obj)

        logger.info(f"Exported subgraph: {len(nodes)} nodes, {len(edges)} edges")

        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'seed_entities': len(seed_entity_ids),
                'retrieved_chunks': len(retrieved_chunk_ids)
            }
        }
