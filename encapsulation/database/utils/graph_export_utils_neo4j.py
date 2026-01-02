"""
Graph Export Utilities for Visualization (Neo4j Version)

Provides utilities to export graph data from PrunedHippoRAGNeo4jStore
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


def _kg_schema_status(graph_store) -> tuple[bool, str | None, str | None]:
    schema = getattr(graph_store, "kg_schema", None)
    loaded = schema is not None
    version = getattr(schema, "version", None) if loaded else None
    config = getattr(graph_store, "config", None)
    schema_path = getattr(config, "kg_schema_path", None) if config is not None else None
    return loaded, (str(version) if version is not None else None), (str(schema_path) if schema_path else None)


def _kg_ingest_stats(graph_store, *, owner_key: str | None) -> Dict[str, Any] | None:
    if not owner_key:
        return None
    try:
        rows = graph_store._execute_query(
            """
            MATCH (m:KGIngestMeta {owner_id: $owner_id})
            RETURN m.triples_total AS triples_total,
                   m.triples_kept AS triples_kept,
                   m.triples_dropped_endpoints AS triples_dropped_endpoints,
                   m.triples_dropped_ambiguous_endpoints AS triples_dropped_ambiguous_endpoints,
                   m.triples_dropped_schema AS triples_dropped_schema,
                   m.endpoint_drop_ratio AS endpoint_drop_ratio,
                   m.fact_provenance_max_source_chunks AS fact_provenance_max_source_chunks,
                   m.updated_at AS updated_at
            LIMIT 1
            """,
            {"owner_id": str(owner_key)},
        )
        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        if not row0:
            return None
        updated_at = row0.get("updated_at")
        return {
            "triples_total": int(row0.get("triples_total") or 0),
            "triples_kept": int(row0.get("triples_kept") or 0),
            "triples_dropped_endpoints": int(row0.get("triples_dropped_endpoints") or 0),
            "triples_dropped_ambiguous_endpoints": int(row0.get("triples_dropped_ambiguous_endpoints") or 0),
            "triples_dropped_schema": int(row0.get("triples_dropped_schema") or 0),
            "endpoint_drop_ratio": float(row0.get("endpoint_drop_ratio") or 0.0),
            "fact_provenance_max_source_chunks": row0.get("fact_provenance_max_source_chunks"),
            "updated_at": str(updated_at) if updated_at is not None else None,
        }
    except Exception:
        return None


def _edge_directed_by_schema(graph_store, *, predicate: str, domain: str | None, force_directed: bool) -> bool:
    if force_directed:
        return True
    schema = getattr(graph_store, "kg_schema", None)
    if schema is None:
        return False
    try:
        domain_schema = schema.for_domain(str(domain or "").strip() or None)
        return bool(domain_schema.is_direction_sensitive(str(predicate or "")))
    except Exception:
        return False


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
        include_node_types: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
        owner_scope_label: Optional[str] = None,
        *,
        directed_edges: bool = False,
        preserve_multi_edges: bool = False,
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

        owner_key = None
        if owner_id is not None:
            owner_key = graph_store._owner_key(owner_id)

        chunks = []
        nodes = []  # Only entities
        edges = []
        categories_set = set()  # Track unique entity types

        # Get node statistics
        params = {'global_owner': graph_store.OWNER_GLOBAL_KEY}
        owner_clause = ""
        if owner_key is not None:
            owner_clause = "AND COALESCE(n.owner_id, $global_owner) = $owner_id"
            params['owner_id'] = owner_key

        count_query = f"""
        MATCH (n)
        WHERE n:Chunk OR n:Entity
          {owner_clause}
        RETURN count(n) AS total_nodes
        """
        result = graph_store._execute_query(count_query, params)
        total_nodes = result[0]['total_nodes'] if result else 0
        logger.info(f"Exporting graph with {total_nodes} nodes")

        # Sample nodes if too many
        if total_nodes > max_nodes:
            logger.warning(f"Graph has {total_nodes} nodes, sampling {max_nodes} nodes")
            # Sample nodes by degree (keep high-degree nodes)
            # Use COUNT {} instead of size() for Neo4j 5.x compatibility
            node_query = f"""
            MATCH (n)
            WHERE n:Chunk OR n:Entity
              {owner_clause}
            WITH n, COUNT {{ (n)--() }} AS degree
            ORDER BY degree DESC
            LIMIT $max_nodes
            RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
                   CASE WHEN n:Chunk THEN 'chunk' ELSE 'entity' END AS node_type,
                   n.content AS content,
                   n.entity_name AS entity_name,
                   n.entity_type AS entity_type
            """
            sampled_nodes = graph_store._execute_query(node_query, {**params, 'max_nodes': max_nodes})
        else:
            node_query = f"""
            MATCH (n)
            WHERE n:Chunk OR n:Entity
              {owner_clause}
            RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
                   CASE WHEN n:Chunk THEN 'chunk' ELSE 'entity' END AS node_type,
                   n.content AS content,
                   n.entity_name AS entity_name,
                   n.entity_type AS entity_type
            """
            sampled_nodes = graph_store._execute_query(node_query, params)

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
                    chunk_obj['content'] = _truncate_text(
                        record['content'],
                        limit=int(GRAPH_EXPORT_CHUNK_CONTENT_PREVIEW_CHARS),
                    )
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
        edge_clause = ""
        edge_params = {'node_ids': list(node_set), 'global_owner': graph_store.OWNER_GLOBAL_KEY}
        if owner_key is not None:
            edge_clause = """
          AND COALESCE(n1.owner_id, $global_owner) = $owner_id
          AND COALESCE(n2.owner_id, $global_owner) = $owner_id
          AND COALESCE(r.owner_id, $global_owner) = $owner_id
            """
            edge_params['owner_id'] = owner_key

        edge_results = []
        max_edges = max(0, int(max_edges or 0))
        if max_edges > 0:
            factor = max(1, int(GRAPH_EXPORT_EDGE_FETCH_FACTOR))
            fetch_limit = min(int(GRAPH_EXPORT_EDGE_FETCH_MAX), max_edges * factor)
            edge_query = f"""
            MATCH (n1)-[r]-(n2)
            WHERE (COALESCE(n1.chunk_id, n1.entity_id) IN $node_ids)
              AND (COALESCE(n2.chunk_id, n2.entity_id) IN $node_ids)
              {edge_clause}
            WITH startNode(r) AS s, endNode(r) AS t, r AS r
            RETURN COALESCE(s.chunk_id, s.entity_id) AS source_id,
                   COALESCE(t.chunk_id, t.entity_id) AS target_id,
                   type(r) AS rel_type,
                   COALESCE(r.weight, r.similarity, 1.0) AS weight,
                   r.predicate AS predicate,
                   r.fact_id AS fact_id,
                   r.source_chunk_ids AS source_chunk_ids,
                   r.source_chunk_ids_truncated AS source_chunk_ids_truncated,
                   r.occurrences AS occurrences,
                   r.schema_version AS schema_version,
                   r.domain AS domain,
                   CASE WHEN s:Chunk THEN 'chunk' ELSE 'entity' END AS source_type,
                   CASE WHEN t:Chunk THEN 'chunk' ELSE 'entity' END AS target_type,
                   s.entity_name AS source_name,
                   t.entity_name AS target_name
            ORDER BY weight DESC
            LIMIT $edge_limit
            """
            edge_params = dict(edge_params)
            edge_params["edge_limit"] = int(fetch_limit)
            edge_results = graph_store._execute_query(edge_query, edge_params)

        kg_schema_loaded, kg_schema_version, kg_schema_path = _kg_schema_status(graph_store)
        kg_ingest_stats = _kg_ingest_stats(graph_store, owner_key=owner_key)

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

            predicate_value = str(record.get("predicate") or "").strip()
            domain_value = str(record.get("domain") or "").strip() or None
            edge_directed = _edge_directed_by_schema(
                graph_store,
                predicate=predicate_value,
                domain=domain_value,
                force_directed=bool(directed_edges),
            )

            # Avoid duplicate edges.
            # When `preserve_multi_edges` is enabled (debug/audit), do not collapse different predicates
            # between the same node pair in undirected exports.
            if edge_directed:
                edge_key = str(record.get("fact_id") or f"{source_id}->{target_id}:{rel_type}:{predicate_value}")
            elif preserve_multi_edges:
                a, b = sorted([source_id, target_id])
                edge_key = (a, b, rel_type, predicate_value)
            else:
                edge_key = tuple(sorted([source_id, target_id]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Build edge object
            edge_obj = {
                'id': str(record.get("fact_id") or f"{source_id}_{target_id}_{rel_type}_{predicate_value or 'related'}"),
                'weight': weight
            }

            # Determine relation and source/target display
            if source_type == 'chunk' and target_type == 'entity':
                edge_obj['source'] = source_id  # chunk_id
                edge_obj['target'] = record.get('target_name') or target_id  # entity_name or entity_id
                edge_obj['relation'] = 'mentions'
                edge_obj['directed'] = True
                edges_by_type['mentions'].append(edge_obj)

            elif source_type == 'entity' and target_type == 'chunk':
                # Skip reverse edge (mentioned_by), as edges are undirected
                continue

            elif source_type == 'entity' and target_type == 'entity':
                source_name = record.get('source_name') or source_id
                target_name = record.get('target_name') or target_id

                if rel_type == 'RELATES_TO':
                    # Fact relation - use predicate from edge
                    predicate = predicate_value or 'related'
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = predicate
                    edge_obj['directed'] = bool(edge_directed)
                    edge_obj['fact_id'] = record.get("fact_id")
                    edge_obj['source_id'] = source_id
                    edge_obj['target_id'] = target_id
                    edge_obj['source_chunk_ids'] = record.get("source_chunk_ids") or []
                    edge_obj['source_chunk_ids_truncated'] = bool(record.get("source_chunk_ids_truncated") or False)
                    edge_obj['occurrences'] = int(record.get("occurrences") or 0) if record.get("occurrences") is not None else None
                    edge_obj['schema_version'] = record.get("schema_version")
                    edge_obj['domain'] = domain_value
                    edges_by_type['fact_relation'].append(edge_obj)
                else:
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = 'related'
                    edge_obj['directed'] = False
                    edges_by_type['other'].append(edge_obj)
            else:
                edge_obj['source'] = source_id
                edge_obj['target'] = target_id
                edge_obj['relation'] = 'related'
                edge_obj['directed'] = False
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
        edge_count_clause = ""
        edge_count_params = {'global_owner': graph_store.OWNER_GLOBAL_KEY}
        if owner_key is not None:
            edge_count_clause = """
        WHERE COALESCE(n1.owner_id, $global_owner) = $owner_id
          AND COALESCE(n2.owner_id, $global_owner) = $owner_id
          AND COALESCE(r.owner_id, $global_owner) = $owner_id
            """
            edge_count_params['owner_id'] = owner_key

        edge_count_query = f"""
        MATCH (n1)-[r]-(n2)
        {edge_count_clause}
        RETURN count(r)/2 AS total_edges
        """
        edge_count_result = graph_store._execute_query(edge_count_query, edge_count_params)
        total_edges = edge_count_result[0]['total_edges'] if edge_count_result else 0

        # Build categories list from unique entity types
        categories = [{'name': cat} for cat in sorted(categories_set)]

        scope_value = owner_scope_label if owner_scope_label is not None else owner_key

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
                'categories': categories,
                'owner_scope': scope_value,
                'directed_edges': bool(directed_edges),
                'preserve_multi_edges': bool(preserve_multi_edges),
                'kg_schema_loaded': bool(kg_schema_loaded),
                'kg_schema_version': kg_schema_version,
                'kg_schema_path': kg_schema_path,
                'kg_ingest_stats': kg_ingest_stats,
            }
        }
    
    @staticmethod
    def export_subgraph(
        graph_store,
        subgraph_node_ids: Set[str],
        seed_entity_ids: Optional[Set[str]] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        node_ppr_scores: Optional[Dict[str, float]] = None,
        *,
        max_edges: int = 2000,
        owner_id: Optional[str] = None,
        owner_scope_label: Optional[str] = None,
        directed_edges: bool = False,
        preserve_multi_edges: bool = False,
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
        chunks: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []  # Only entities
        edges: List[Dict[str, Any]] = []
        categories_set = set()  # Track unique entity types

        owner_key = None
        if owner_id is not None:
            owner_key = graph_store._owner_key(owner_id)
        global_owner = graph_store.OWNER_GLOBAL_KEY

        kg_schema_loaded, kg_schema_version, kg_schema_path = _kg_schema_status(graph_store)
        kg_ingest_stats = _kg_ingest_stats(graph_store, owner_key=owner_key)

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
                    'categories': [],
                    'directed_edges': bool(directed_edges),
                    'preserve_multi_edges': bool(preserve_multi_edges),
                    'kg_schema_loaded': bool(kg_schema_loaded),
                    'kg_schema_version': kg_schema_version,
                    'kg_schema_path': kg_schema_path,
                    'kg_ingest_stats': kg_ingest_stats,
                }
            }

        # Prefer indexed lookups over `OR` clauses to keep Neo4j exports fast and stable under load.
        normalized_ids = [str(node_id) for node_id in subgraph_node_ids if str(node_id)]
        entity_ids = [node_id for node_id in normalized_ids if node_id.startswith("entity-")]
        chunk_ids = [node_id for node_id in normalized_ids if not node_id.startswith("entity-")]

        node_results: List[Dict[str, Any]] = []
        if chunk_ids:
            clause = ""
            params: Dict[str, Any] = {"chunk_ids": chunk_ids, "global_owner": global_owner}
            if owner_key is not None:
                clause = "AND COALESCE(c.owner_id, $global_owner) = $owner_id"
                params["owner_id"] = owner_key
            node_results.extend(
                graph_store._execute_query(
                    f"""
                    MATCH (c:Chunk)
                    WHERE c.chunk_id IN $chunk_ids
                      {clause}
                    RETURN c.chunk_id AS node_id,
                           'chunk' AS node_type,
                           c.content AS content,
                           NULL AS entity_name,
                           NULL AS entity_type
                    """,
                    params,
                )
            )
        if entity_ids:
            clause = ""
            params = {"entity_ids": entity_ids, "global_owner": global_owner}
            if owner_key is not None:
                clause = "AND COALESCE(e.owner_id, $global_owner) = $owner_id"
                params["owner_id"] = owner_key
            node_results.extend(
                graph_store._execute_query(
                    f"""
                    MATCH (e:Entity)
                    WHERE e.entity_id IN $entity_ids
                      {clause}
                    RETURN e.entity_id AS node_id,
                           'entity' AS node_type,
                           NULL AS content,
                           e.entity_name AS entity_name,
                           e.entity_type AS entity_type
                    """,
                    params,
                )
            )

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

        max_edges = max(0, int(max_edges or 0))
        edge_results: List[Dict[str, Any]] = []

        edge_params: Dict[str, Any] = {"global_owner": global_owner}
        rel_owner_clause = ""
        if owner_key is not None:
            edge_params["owner_id"] = owner_key
            rel_owner_clause = "AND (r.owner_id IS NULL OR COALESCE(r.owner_id, $global_owner) = $owner_id)"

        edge_clause_chunk_entity = ""
        edge_clause_entity_entity = ""
        if owner_key is not None:
            edge_clause_chunk_entity = (
                "AND COALESCE(c.owner_id, $global_owner) = $owner_id "
                "AND COALESCE(e.owner_id, $global_owner) = $owner_id "
                + rel_owner_clause
            )
            edge_clause_entity_entity = (
                "AND COALESCE(e1.owner_id, $global_owner) = $owner_id "
                "AND COALESCE(e2.owner_id, $global_owner) = $owner_id "
                + rel_owner_clause
            )

        # Chunk-Entity edges
        if chunk_ids and entity_ids:
            params = dict(edge_params)
            params.update({"chunk_ids": chunk_ids, "entity_ids": entity_ids, "limit": max_edges or 2000})
            edge_results.extend(
                graph_store._execute_query(
                    f"""
                    MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
                    WHERE c.chunk_id IN $chunk_ids
                      AND e.entity_id IN $entity_ids
                      {edge_clause_chunk_entity}
                    RETURN c.chunk_id AS source_id,
                           e.entity_id AS target_id,
                           type(r) AS rel_type,
                           COALESCE(r.weight, r.similarity, 1.0) AS weight,
                           r.predicate AS predicate,
                           r.fact_id AS fact_id,
                           r.source_chunk_ids AS source_chunk_ids,
                           r.source_chunk_ids_truncated AS source_chunk_ids_truncated,
                           r.occurrences AS occurrences,
                           r.schema_version AS schema_version,
                           r.domain AS domain,
                           'chunk' AS source_type,
                           'entity' AS target_type,
                           NULL AS source_name,
                           e.entity_name AS target_name
                    ORDER BY weight DESC
                    LIMIT $limit
                    """,
                    params,
                )
            )

        # Entity-Entity edges
        if entity_ids:
            params = dict(edge_params)
            params.update({"entity_ids": entity_ids, "limit": max_edges or 2000})
            # Always fetch directed fact edges so direction-sensitive predicates can preserve direction,
            # while non-sensitive predicates can still be rendered as undirected via schema.
            edge_results.extend(
                graph_store._execute_query(
                    f"""
                    MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
                    WHERE e1.entity_id IN $entity_ids
                      AND e2.entity_id IN $entity_ids
                      {edge_clause_entity_entity}
                    RETURN e1.entity_id AS source_id,
                           e2.entity_id AS target_id,
                           type(r) AS rel_type,
                           COALESCE(r.weight, 1.0) AS weight,
                           r.predicate AS predicate,
                           r.fact_id AS fact_id,
                           r.source_chunk_ids AS source_chunk_ids,
                           r.source_chunk_ids_truncated AS source_chunk_ids_truncated,
                           r.occurrences AS occurrences,
                           r.schema_version AS schema_version,
                           r.domain AS domain,
                           'entity' AS source_type,
                           'entity' AS target_type,
                           e1.entity_name AS source_name,
                           e2.entity_name AS target_name
                    ORDER BY weight DESC
                    LIMIT $limit
                    """,
                    params,
                )
            )

        # Export edges
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

            predicate_value = str(record.get("predicate") or "").strip()
            domain_value = str(record.get("domain") or "").strip() or None
            edge_directed = _edge_directed_by_schema(
                graph_store,
                predicate=predicate_value,
                domain=domain_value,
                force_directed=bool(directed_edges),
            )
            if edge_directed:
                edge_key = str(record.get("fact_id") or f"{source_id}->{target_id}:{rel_type}:{predicate_value}")
            elif preserve_multi_edges:
                a, b = sorted([source_id, target_id])
                edge_key = (a, b, rel_type, predicate_value)
            else:
                edge_key = tuple(sorted([source_id, target_id]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Build edge object
            edge_obj = {
                'id': str(record.get("fact_id") or f"{source_id}_{target_id}_{rel_type}_{predicate_value or 'related'}"),
                'weight': weight
            }

            # Determine relation and source/target display
            if source_type == 'chunk' and target_type == 'entity':
                # Chunk mentions entity
                edge_obj['source'] = source_id  # chunk_id
                edge_obj['target'] = record.get('target_name') or target_id  # entity_name or entity_id
                edge_obj['relation'] = 'mentions'
                edge_obj['directed'] = True

            elif source_type == 'entity' and target_type == 'chunk':
                # Skip reverse edge (mentioned_by), as edges are undirected
                continue

            elif source_type == 'entity' and target_type == 'entity':
                # Entity-entity relation (fact-based)
                source_name = record.get('source_name') or source_id
                target_name = record.get('target_name') or target_id

                if rel_type == 'RELATES_TO':
                    # Fact relation - use predicate from edge
                    predicate = predicate_value or 'related'
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = predicate
                    edge_obj['directed'] = bool(edge_directed)
                    edge_obj['fact_id'] = record.get("fact_id")
                    edge_obj['source_id'] = source_id
                    edge_obj['target_id'] = target_id
                    edge_obj['source_chunk_ids'] = record.get("source_chunk_ids") or []
                    edge_obj['source_chunk_ids_truncated'] = bool(record.get("source_chunk_ids_truncated") or False)
                    edge_obj['occurrences'] = int(record.get("occurrences") or 0) if record.get("occurrences") is not None else None
                    edge_obj['schema_version'] = record.get("schema_version")
                    edge_obj['domain'] = domain_value
                else:
                    edge_obj['source'] = source_name
                    edge_obj['target'] = target_name
                    edge_obj['relation'] = 'related'
                    edge_obj['directed'] = False

            else:
                # Fallback for other edge types
                edge_obj['source'] = source_id
                edge_obj['target'] = target_id
                edge_obj['relation'] = 'related'
                edge_obj['directed'] = False

            edges.append(edge_obj)

        if max_edges and len(edges) > max_edges:
            edges = edges[:max_edges]

        logger.info(
            "Exported subgraph: %d chunks, %d entities, %d edges (owner_scope=%s)",
            len(chunks),
            len(nodes),
            len(edges),
            owner_scope_label if owner_scope_label is not None else owner_key,
        )

        # Build categories list from unique entity types
        categories = [{'name': cat} for cat in sorted(categories_set)]

        scope_value = owner_scope_label if owner_scope_label is not None else owner_key
        return {
            'chunks': chunks,
            'nodes': nodes,  # Only entities
            'edges': edges,
            'metadata': {
                'total_nodes': len(chunks) + len(nodes),
                'total_edges': len(edges),
                'seed_entities': len(seed_entity_ids),
                'retrieved_chunks': len(retrieved_chunk_ids),
                'categories': categories,
                'owner_scope': scope_value,
                'directed_edges': bool(directed_edges),
                'preserve_multi_edges': bool(preserve_multi_edges),
                'kg_schema_loaded': bool(kg_schema_loaded),
                'kg_schema_version': kg_schema_version,
                'kg_schema_path': kg_schema_path,
                'kg_ingest_stats': kg_ingest_stats,
            }
        }
