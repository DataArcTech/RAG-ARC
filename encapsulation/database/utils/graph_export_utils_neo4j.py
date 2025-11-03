"""
Graph Export Utilities for Visualization (Neo4j Version)

Provides utilities to export graph data from PrunedHippoRAGNeo4jStore
for frontend visualization using Cytoscape.js or other graph libraries.
"""

import logging
import re
from typing import Dict, List, Any, Set, Optional

logger = logging.getLogger(__name__)


class GraphExporterNeo4j:
    """Export graph data from Neo4j for visualization"""

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
        include_node_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export complete graph for visualization

        Args:
            graph_store: PrunedHippoRAGNeo4jStore instance
            max_nodes: Maximum number of nodes to export (for performance)
            max_edges: Maximum number of edges to export
            include_node_types: List of node types to include ['chunk', 'entity']
                               If None, include all types

        Returns:
            Dict with 'chunks', 'nodes' (entities), 'edges', and 'metadata'
        """
        if include_node_types is None:
            include_node_types = ['chunk', 'entity']

        chunks = []
        nodes = []  # Only entities
        edges = []
        categories_set = set()  # Track unique entity types

        # Get node statistics
        count_query = """
        MATCH (n)
        WHERE n:Chunk OR n:Entity
        RETURN count(n) AS total_nodes
        """
        result = graph_store._execute_query(count_query)
        total_nodes = result[0]['total_nodes'] if result else 0
        logger.info(f"Exporting graph with {total_nodes} nodes")

        # Sample nodes if too many
        if total_nodes > max_nodes:
            logger.warning(f"Graph has {total_nodes} nodes, sampling {max_nodes} nodes")
            # Sample nodes by degree (keep high-degree nodes)
            # Use COUNT {} instead of size() for Neo4j 5.x compatibility
            node_query = """
            MATCH (n)
            WHERE n:Chunk OR n:Entity
            WITH n, COUNT { (n)--() } AS degree
            ORDER BY degree DESC
            LIMIT $max_nodes
            RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
                   CASE WHEN n:Chunk THEN 'chunk' ELSE 'entity' END AS node_type,
                   n.content AS content,
                   n.entity_name AS entity_name,
                   n.entity_type AS entity_type
            """
            sampled_nodes = graph_store._execute_query(node_query, {'max_nodes': max_nodes})
        else:
            node_query = """
            MATCH (n)
            WHERE n:Chunk OR n:Entity
            RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
                   CASE WHEN n:Chunk THEN 'chunk' ELSE 'entity' END AS node_type,
                   n.content AS content,
                   n.entity_name AS entity_name,
                   n.entity_type AS entity_type
            """
            sampled_nodes = graph_store._execute_query(node_query)

        # Export nodes - separate chunks and entities
        node_set = set()
        for record in sampled_nodes:
            node_id = record['node_id']
            node_type = record['node_type']

            # Filter by node type
            if node_type not in include_node_types:
                continue

            node_set.add(node_id)

            # Build node structure based on type
            if node_type == 'chunk':
                chunk_obj = {
                    'id': node_id,
                    'type': 'chunk'
                }
                if record.get('content'):
                    chunk_obj['content'] = record['content']
                chunks.append(chunk_obj)

            elif node_type == 'entity':
                entity_name = record.get('entity_name')

                # Filter out entities that are pure numbers, timestamps, or time formats
                if GraphExporterNeo4j._should_filter_entity(entity_name):
                    continue

                entity_obj = {
                    'id': node_id,
                    'type': 'entity'
                }
                # Use 'name' instead of 'entity_name' or 'entity_text'
                if entity_name:
                    entity_obj['name'] = entity_name
                # Use 'category' for entity_type
                entity_type = record.get('entity_type', 'Entity')
                entity_obj['category'] = entity_type
                categories_set.add(entity_type)

                nodes.append(entity_obj)
        
        # Collect edges by type for uniform sampling
        edges_by_type = {
            'mentions': [],
            'fact_relation': [],
            'synonymy': [],
            'other': []
        }
        
        # Query edges between sampled nodes
        edge_query = """
        MATCH (n1)-[r]-(n2)
        WHERE (COALESCE(n1.chunk_id, n1.entity_id) IN $node_ids)
          AND (COALESCE(n2.chunk_id, n2.entity_id) IN $node_ids)
        RETURN COALESCE(n1.chunk_id, n1.entity_id) AS source_id,
               COALESCE(n2.chunk_id, n2.entity_id) AS target_id,
               type(r) AS rel_type,
               COALESCE(r.weight, r.similarity, 1.0) AS weight,
               r.predicate AS predicate,
               CASE WHEN n1:Chunk THEN 'chunk' ELSE 'entity' END AS source_type,
               CASE WHEN n2:Chunk THEN 'chunk' ELSE 'entity' END AS target_type,
               n1.entity_name AS source_name,
               n2.entity_name AS target_name
        """
        edge_results = graph_store._execute_query(edge_query, {'node_ids': list(node_set)})

        seen_edges = set()
        for record in edge_results:
            source_id = record['source_id']
            target_id = record['target_id']
            rel_type = record['rel_type']
            weight = record['weight']
            source_type = record['source_type']
            target_type = record['target_type']

            # Skip if not in sampled nodes
            if source_id not in node_set or target_id not in node_set:
                continue

            # Skip SIMILAR_TO relationships
            if rel_type == 'SIMILAR_TO':
                continue

            # Avoid duplicate edges (undirected)
            edge_key = tuple(sorted([source_id, target_id]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Build edge object
            edge_obj = {
                'id': f"{source_id}_{target_id}",
                'weight': weight
            }

            # Determine relation and source/target display
            if source_type == 'chunk' and target_type == 'entity':
                edge_obj['source'] = source_id  # chunk_id
                edge_obj['target'] = record.get('target_name') or target_id  # entity_name or entity_id
                edge_obj['relation'] = 'mentions'
                edges_by_type['mentions'].append(edge_obj)

            elif source_type == 'entity' and target_type == 'chunk':
                # Skip reverse edge (mentioned_by), as edges are undirected
                continue

            elif source_type == 'entity' and target_type == 'entity':
                source_name = record.get('source_name') or source_id
                target_name = record.get('target_name') or target_id

                if rel_type == 'RELATES_TO':
                    # Fact relation - use predicate from edge
                    predicate = record.get('predicate') or 'related'
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = predicate
                    edges_by_type['fact_relation'].append(edge_obj)
                else:
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = 'related'
                    edges_by_type['other'].append(edge_obj)
            else:
                edge_obj['source'] = source_id
                edge_obj['target'] = target_id
                edge_obj['relation'] = 'related'
                edges_by_type['other'].append(edge_obj)
        
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

        # Get total edge count
        edge_count_query = "MATCH ()-[r]-() RETURN count(r)/2 AS total_edges"
        edge_count_result = graph_store._execute_query(edge_count_query)
        total_edges = edge_count_result[0]['total_edges'] if edge_count_result else 0

        # Build categories list from unique entity types
        categories = [{'name': cat} for cat in sorted(categories_set)]

        return {
            'chunks': chunks,
            'nodes': nodes,  # Only entities
            'edges': edges,
            'metadata': {
                'total_nodes': total_nodes,
                'total_edges': int(total_edges),
                'exported_nodes': len(chunks) + len(nodes),
                'exported_edges': len(edges),
                'sampled': total_nodes > max_nodes,
                'categories': categories
            }
        }
    
    @staticmethod
    def export_subgraph(
        graph_store,
        subgraph_node_ids: Set[str],
        seed_entity_ids: Optional[Set[str]] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        node_ppr_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Export retrieval subgraph for visualization (Neo4j version)

        Args:
            graph_store: PrunedHippoRAGNeo4jStore instance
            subgraph_node_ids: Set of node IDs in the subgraph (chunk_ids or entity_ids)
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

        seed_entity_ids = seed_entity_ids or set()
        retrieved_chunk_ids = retrieved_chunk_ids or []
        node_ppr_scores = node_ppr_scores or {}

        if not subgraph_node_ids:
            logger.warning("No subgraph nodes provided")
            return {
                'chunks': [],
                'nodes': [],
                'edges': [],
                'metadata': {
                    'total_nodes': 0,
                    'total_edges': 0,
                    'seed_entities': len(seed_entity_ids),
                    'retrieved_chunks': len(retrieved_chunk_ids),
                    'categories': []
                }
            }

        # Query nodes from Neo4j
        node_query = """
        MATCH (n)
        WHERE (n.chunk_id IN $node_ids OR n.entity_id IN $node_ids)
        RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
               CASE WHEN n:Chunk THEN 'chunk' ELSE 'entity' END AS node_type,
               n.content AS content,
               n.entity_name AS entity_name,
               n.entity_type AS entity_type
        """
        node_results = graph_store._execute_query(node_query, {'node_ids': list(subgraph_node_ids)})

        # Export nodes - separate chunks and entities
        seed_marked_count = 0
        for record in node_results:
            node_id = record['node_id']
            node_type = record['node_type']

            # Build node structure based on type
            if node_type == 'chunk':
                chunk_obj = {
                    'id': node_id,
                    'type': 'chunk'
                }
                if record.get('content'):
                    chunk_obj['content'] = record['content']
                # Add PPR score if available
                if node_ppr_scores and node_id in node_ppr_scores:
                    chunk_obj['ppr_score'] = node_ppr_scores[node_id]
                chunks.append(chunk_obj)

            elif node_type == 'entity':
                entity_name = record.get('entity_name')

                # Filter out entities that are pure numbers, timestamps, or time formats
                if GraphExporterNeo4j._should_filter_entity(entity_name):
                    continue

                entity_obj = {
                    'id': node_id,
                    'type': 'entity'
                }
                # Use 'name' instead of 'entity_name' or 'entity_text'
                if entity_name:
                    entity_obj['name'] = entity_name
                # Use 'category' for entity_type
                entity_type = record.get('entity_type', 'Entity')
                entity_obj['category'] = entity_type
                categories_set.add(entity_type)

                # Mark seed entities (check if node_id is in seed_entity_ids set)
                if seed_entity_ids and node_id in seed_entity_ids:
                    entity_obj['is_seed'] = True
                    seed_marked_count += 1

                # Add PPR score if available
                if node_ppr_scores and node_id in node_ppr_scores:
                    entity_obj['ppr_score'] = node_ppr_scores[node_id]

                nodes.append(entity_obj)

        if seed_entity_ids:
            logger.info(f"Marked {seed_marked_count} seed entities out of {len(seed_entity_ids)} provided")

        # Query edges within subgraph
        edge_query = """
        MATCH (n1)-[r]-(n2)
        WHERE (n1.chunk_id IN $node_ids OR n1.entity_id IN $node_ids)
          AND (n2.chunk_id IN $node_ids OR n2.entity_id IN $node_ids)
        RETURN COALESCE(n1.chunk_id, n1.entity_id) AS source_id,
               COALESCE(n2.chunk_id, n2.entity_id) AS target_id,
               type(r) AS rel_type,
               COALESCE(r.weight, r.similarity, 1.0) AS weight,
               r.predicate AS predicate,
               CASE WHEN n1:Chunk THEN 'chunk' ELSE 'entity' END AS source_type,
               CASE WHEN n2:Chunk THEN 'chunk' ELSE 'entity' END AS target_type,
               n1.entity_name AS source_name,
               n2.entity_name AS target_name
        """
        edge_results = graph_store._execute_query(edge_query, {'node_ids': list(subgraph_node_ids)})

        # Export edges (skip reverse edges to avoid duplicates)
        seen_edges = set()
        for record in edge_results:
            source_id = record['source_id']
            target_id = record['target_id']
            rel_type = record['rel_type']
            weight = record['weight']
            source_type = record['source_type']
            target_type = record['target_type']

            # Skip SIMILAR_TO relationships
            if rel_type == 'SIMILAR_TO':
                continue

            # Avoid duplicate edges (undirected)
            edge_key = tuple(sorted([source_id, target_id]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Build edge object
            edge_obj = {
                'id': f"{source_id}_{target_id}",
                'weight': weight
            }

            # Determine relation and source/target display
            if source_type == 'chunk' and target_type == 'entity':
                # Chunk mentions entity
                edge_obj['source'] = source_id  # chunk_id
                edge_obj['target'] = record.get('target_name') or target_id  # entity_name or entity_id
                edge_obj['relation'] = 'mentions'

            elif source_type == 'entity' and target_type == 'chunk':
                # Skip reverse edge (mentioned_by), as edges are undirected
                continue

            elif source_type == 'entity' and target_type == 'entity':
                # Entity-entity relation (fact-based)
                source_name = record.get('source_name') or source_id
                target_name = record.get('target_name') or target_id

                if rel_type == 'RELATES_TO':
                    # Fact relation - use predicate from edge
                    predicate = record.get('predicate') or 'related'
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = predicate
                else:
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = 'related'

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

