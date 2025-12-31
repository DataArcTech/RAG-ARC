import logging
from typing import List, Dict, Any, Optional, Set, Tuple

import igraph as ig
import warnings

warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPy.* has no __module__ attribute",
    category=DeprecationWarning,
)

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jCacheMixin:
    def get_neighbors_with_weights(self, node_id: str, owner_id: Optional[Any] = None) -> List[Tuple[str, float]]:
        """
        Get all neighbors of a node with their edge weights from Neo4j.

        Args:
            node_id: Node ID (chunk_id or entity_id)
            owner_id: Owner scope for the query

        Returns:
            List of (neighbor_id, weight) tuples
        """
        all_owners = owner_id is None
        owner_key = None if all_owners else self._owner_key(owner_id)

        with self.read_lock():
            if self._cache_loaded and self._graph_cache is not None:
                if all_owners:
                    aggregated: list[tuple[str, float]] = []
                    for shard in self._graph_cache.values():
                        aggregated.extend(shard.get(node_id, ()))
                    return list(aggregated)
                owner_neighbors = self._graph_cache.get(owner_key, {})
                return list(owner_neighbors.get(node_id, ()))

        # Optimized query: use single MATCH with OR condition
        if all_owners:
            query = """
            MATCH (n)-[r]-(neighbor)
            WHERE (n.chunk_id = $node_id OR n.entity_id = $node_id)
            RETURN COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
                   COALESCE(r.weight, r.similarity, 1.0) AS weight
            """
            params = {'node_id': node_id}
        else:
            query = """
            MATCH (n)-[r]-(neighbor)
            WHERE (n.chunk_id = $node_id OR n.entity_id = $node_id)
              AND COALESCE(n.owner_id, $global_owner) = $owner_id
              AND COALESCE(neighbor.owner_id, $global_owner) = $owner_id
              AND COALESCE(r.owner_id, $global_owner) = $owner_id
            RETURN COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
                   COALESCE(r.weight, r.similarity, 1.0) AS weight
            """
            params = {
                'node_id': node_id,
                'owner_id': owner_key,
                'global_owner': self.OWNER_GLOBAL_KEY
            }

        results = self._execute_query(query, params)

        neighbors = []
        for record in results:
            neighbor_id = record['neighbor_id']
            weight = record['weight'] or 1.0
            if neighbor_id:
                neighbors.append((neighbor_id, float(weight)))

        return neighbors

    def _build_entity_chunk_count_cache(self):
        """
        Build entity chunk count cache from graph cache.

        This computes how many chunks each entity appears in, which is used
        for normalizing entity weights during PPR.
        """
        logger.info("Building entity chunk count cache from graph cache...")
        import time
        start_time = time.time()

        entity_chunk_count_cache: Dict[str, Dict[str, int]] = {}

        with self.read_lock():
            if not self._graph_cache:
                logger.warning("Graph cache not loaded, cannot build entity chunk count cache")
                return

            for owner_key, adjacency in self._graph_cache.items():
                owner_counts: Dict[str, int] = {}
                for entity_id, neighbors in adjacency.items():
                    # Only process entity nodes
                    if not entity_id.startswith("entity-"):
                        continue

                    # Count unique chunk neighbors (chunks don't start with "entity-")
                    chunk_count = sum(1 for neighbor_id, _ in neighbors if not neighbor_id.startswith("entity-"))
                    owner_counts[entity_id] = chunk_count

                entity_chunk_count_cache[owner_key] = owner_counts

        with self.write_lock():
            self._entity_chunk_count_cache = entity_chunk_count_cache

        elapsed = time.time() - start_time
        total_entities = sum(len(counts) for counts in entity_chunk_count_cache.values())
        logger.info(f"Entity chunk count cache built: {total_entities} entities in {elapsed:.2f}s")

    @staticmethod
    def _compute_entity_chunk_count_cache(
        graph_cache: Dict[str, Dict[str, List[Tuple[str, float]]]]
    ) -> Dict[str, Dict[str, int]]:
        entity_chunk_count_cache: Dict[str, Dict[str, int]] = {}
        if not graph_cache:
            return entity_chunk_count_cache

        for owner_key, adjacency in graph_cache.items():
            owner_counts: Dict[str, int] = {}
            for entity_id, neighbors in adjacency.items():
                if not str(entity_id).startswith("entity-"):
                    continue
                chunk_count = sum(1 for neighbor_id, _ in (neighbors or []) if not str(neighbor_id).startswith("entity-"))
                owner_counts[str(entity_id)] = int(chunk_count)
            entity_chunk_count_cache[str(owner_key)] = owner_counts
        return entity_chunk_count_cache

    def get_entity_chunk_count_from_cache(self, entity_id: str, owner_id: Optional[Any] = None) -> int:
        """
        Get the number of chunks an entity appears in from cache.

        Args:
            entity_id: Entity ID

        Returns:
            Number of chunks the entity appears in (0 if not found)
        """
        with self.read_lock():
            if self._entity_chunk_count_cache is None:
                logger.warning("Entity chunk count cache not built, returning 0")
                return 0

            if owner_id is None:
                total = 0
                for owner_counts in self._entity_chunk_count_cache.values():
                    total += owner_counts.get(entity_id, 0)
                return total

            owner_key = self._owner_key(owner_id)
            owner_counts = self._entity_chunk_count_cache.get(owner_key, {})
            return owner_counts.get(entity_id, 0)

    def get_batch_entity_chunk_counts_from_cache(self, entity_ids: List[str], owner_id: Optional[Any] = None) -> Dict[str, int]:
        """
        Get chunk counts for multiple entities from cache.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Dictionary mapping entity IDs to chunk counts
        """
        with self.read_lock():
            if self._entity_chunk_count_cache is None:
                logger.warning("Entity chunk count cache not built, returning empty dict")
                return {}

            if owner_id is None:
                aggregated: Dict[str, int] = {eid: 0 for eid in entity_ids}
                for owner_counts in self._entity_chunk_count_cache.values():
                    for eid in entity_ids:
                        aggregated[eid] += owner_counts.get(eid, 0)
                return aggregated

            owner_key = self._owner_key(owner_id)
            owner_counts = self._entity_chunk_count_cache.get(owner_key, {})
            return {eid: owner_counts.get(eid, 0) for eid in entity_ids}

    def _load_graph_cache(self, force_reload: bool = False):
        """
        Load entire graph structure into memory for fast neighbor lookups.
        This trades memory for speed - loads all edges once at startup.

        Args:
            force_reload: If True, force reload even if cache is already loaded
        """
        with self.read_lock():
            if self._cache_loaded and not force_reload:
                return

        logger.info("Loading graph structure into memory cache...")
        import time
        start_time = time.time()

        # Query all edges in the graph
        query = """
        MATCH (n)-[r]-(neighbor)
        RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
               COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
               COALESCE(r.weight, r.similarity, 1.0) AS weight,
               COALESCE(n.owner_id, $global_owner) AS node_owner_id,
               COALESCE(neighbor.owner_id, $global_owner) AS neighbor_owner_id,
               COALESCE(r.owner_id, $global_owner) AS relation_owner_id
        """

        results = self._execute_query(query, {'global_owner': self.OWNER_GLOBAL_KEY})

        # Build adjacency list
        graph_cache: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
        edge_count = 0
        for record in results:
            node_id = record['node_id']
            neighbor_id = record['neighbor_id']
            weight = record['weight'] or 1.0

            node_owner_id = record.get('node_owner_id') or self.OWNER_GLOBAL_KEY
            neighbor_owner_id = record.get('neighbor_owner_id') or self.OWNER_GLOBAL_KEY
            relation_owner_id = record.get('relation_owner_id') or self.OWNER_GLOBAL_KEY

            if node_id and neighbor_id and node_owner_id == neighbor_owner_id == relation_owner_id:
                owner_cache = graph_cache.setdefault(node_owner_id, {})
                owner_cache.setdefault(node_id, []).append((neighbor_id, float(weight)))
                edge_count += 1

        entity_chunk_count_cache: Dict[str, Dict[str, int]] = {}
        for owner_key, adjacency in graph_cache.items():
            owner_counts: Dict[str, int] = {}
            for entity_id, neighbors in adjacency.items():
                if not entity_id.startswith("entity-"):
                    continue
                owner_counts[entity_id] = sum(
                    1 for neighbor_id, _ in neighbors if not neighbor_id.startswith("entity-")
                )
            entity_chunk_count_cache[owner_key] = owner_counts

        with self.write_lock():
            self._graph_cache = graph_cache
            self._entity_chunk_count_cache = entity_chunk_count_cache
            self._cache_loaded = True
        elapsed = time.time() - start_time
        node_total = sum(len(nodes) for nodes in graph_cache.values())
        logger.info(f"Graph cache loaded: {node_total} nodes, {edge_count} edges in {elapsed:.2f}s")

    def _update_graph_cache_incremental(self, new_chunk_ids: List[str], new_entity_ids: List[str]):
        """
        Incrementally update graph cache with new edges from new chunks and entities.

        Args:
            new_chunk_ids: List of newly added chunk IDs
            new_entity_ids: List of newly added entity IDs
        """
        with self.read_lock():
            cache_ready = bool(self._cache_loaded and self._graph_cache is not None)
        if not cache_ready:
            self._load_graph_cache()
            return

        logger.info(f"Incrementally updating graph cache for {len(new_chunk_ids)} chunks and {len(new_entity_ids)} entities...")
        import time
        start_time = time.time()

        # Query edges involving new nodes
        all_new_node_ids = new_chunk_ids + new_entity_ids
        if not all_new_node_ids:
            return

        query = """
        MATCH (n)-[r]-(neighbor)
        WHERE n.chunk_id IN $node_ids OR n.entity_id IN $node_ids
        RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
               COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
               COALESCE(r.weight, r.similarity, 1.0) AS weight,
               COALESCE(n.owner_id, $global_owner) AS node_owner_id,
               COALESCE(neighbor.owner_id, $global_owner) AS neighbor_owner_id,
               COALESCE(r.owner_id, $global_owner) AS relation_owner_id
        """
        results = self._execute_query(query, {'node_ids': all_new_node_ids, 'global_owner': self.OWNER_GLOBAL_KEY})

        # Update adjacency list
        edge_count = 0
        with self.write_lock():
            if self._graph_cache is None:
                self._graph_cache = {}
            for record in results:
                node_id = record['node_id']
                neighbor_id = record['neighbor_id']
                weight = record['weight'] or 1.0

                node_owner_id = record.get('node_owner_id') or self.OWNER_GLOBAL_KEY
                neighbor_owner_id = record.get('neighbor_owner_id') or self.OWNER_GLOBAL_KEY
                relation_owner_id = record.get('relation_owner_id') or self.OWNER_GLOBAL_KEY

                if node_id and neighbor_id and node_owner_id == neighbor_owner_id == relation_owner_id:
                    owner_cache = self._graph_cache.setdefault(node_owner_id, {})
                    node_neighbors = owner_cache.setdefault(node_id, [])
                    if not any(n == neighbor_id for n, _ in node_neighbors):
                        node_neighbors.append((neighbor_id, float(weight)))
                        edge_count += 1

                    reverse_neighbors = owner_cache.setdefault(neighbor_id, [])
                    if not any(n == node_id for n, _ in reverse_neighbors):
                        reverse_neighbors.append((node_id, float(weight)))

            self._entity_chunk_count_cache = self._compute_entity_chunk_count_cache(self._graph_cache)

        elapsed = time.time() - start_time
        logger.info(f"Graph cache updated: added {edge_count} new edges in {elapsed:.2f}s")

    def get_batch_neighbors_with_weights(self, node_ids: List[str], owner_id: Optional[Any] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get neighbors for multiple nodes in a single query (batch operation).
        Uses in-memory cache if available, otherwise queries Neo4j.

        Args:
            node_ids: List of node IDs

        Returns:
            Dictionary mapping node_id to list of (neighbor_id, weight) tuples
        """
        if not node_ids:
            return {}

        all_owners = owner_id is None
        owner_key = None if all_owners else self._owner_key(owner_id)

        # Use cache if loaded
        with self.read_lock():
            if self._cache_loaded and self._graph_cache is not None:
                neighbors_map = {nid: [] for nid in node_ids}
                if all_owners:
                    for shard in self._graph_cache.values():
                        for nid in node_ids:
                            if nid in shard:
                                neighbors_map[nid].extend(shard.get(nid, ()))
                else:
                    owner_neighbors = self._graph_cache.get(owner_key, {})
                    for nid in node_ids:
                        neighbors_map[nid] = list(owner_neighbors.get(nid, ()))
                return neighbors_map

        # Fallback to Neo4j query
        if all_owners:
            query = """
            UNWIND $node_ids AS nid
            MATCH (n)-[r]-(neighbor)
            WHERE (n.chunk_id = nid OR n.entity_id = nid)
            RETURN nid AS node_id,
                   COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
                   COALESCE(r.weight, r.similarity, 1.0) AS weight
            """
            params = {'node_ids': node_ids}
        else:
            query = """
            UNWIND $node_ids AS nid
            MATCH (n)-[r]-(neighbor)
            WHERE (n.chunk_id = nid OR n.entity_id = nid)
              AND COALESCE(n.owner_id, $global_owner) = $owner_id
              AND COALESCE(neighbor.owner_id, $global_owner) = $owner_id
              AND COALESCE(r.owner_id, $global_owner) = $owner_id
            RETURN nid AS node_id,
                   COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
                   COALESCE(r.weight, r.similarity, 1.0) AS weight
            """
            params = {
                'node_ids': node_ids,
                'owner_id': owner_key,
                'global_owner': self.OWNER_GLOBAL_KEY
            }

        results = self._execute_query(query, params)

        # Group by node_id
        neighbors_map = {nid: [] for nid in node_ids}
        for record in results:
            node_id = record['node_id']
            neighbor_id = record['neighbor_id']
            weight = record['weight'] or 1.0
            if neighbor_id and node_id in neighbors_map:
                neighbors_map[node_id].append((neighbor_id, float(weight)))

        return neighbors_map

    def extract_subgraph_from_cache(self, subgraph_node_ids: Set[str], owner_id: Optional[Any] = None) -> Tuple[ig.Graph, Dict[str, int], Dict[int, str]]:
        """
        Extract a subgraph from in-memory cache and convert to igraph for PageRank computation.

        This method is much faster than extract_subgraph_for_ppr as it avoids Neo4j queries.

        Args:
            subgraph_node_ids: Set of node IDs to include in the subgraph

        Returns:
            Tuple of (igraph, node_to_idx, idx_to_node)
        """
        if not subgraph_node_ids:
            logger.warning("Empty subgraph node set")
            return ig.Graph(directed=False), {}, {}

        with self.read_lock():
            # Require cache to be loaded
            if not (self._cache_loaded and self._graph_cache):
                logger.error("Graph cache not loaded, cannot extract subgraph")
                return ig.Graph(directed=False), {}, {}

            if owner_id is None:
                owner_cache: Dict[str, List[Tuple[str, float]]] = {}
                for shard in self._graph_cache.values():
                    for node_id, neighbors in shard.items():
                        owner_cache.setdefault(node_id, []).extend(neighbors)
            else:
                owner_key = self._owner_key(owner_id)
                owner_cache = self._graph_cache.get(owner_key, {})

        logger.info(f"Extracting subgraph with {len(subgraph_node_ids)} nodes from cache...")

        # Build node mappings
        node_to_idx = {node_id: i for i, node_id in enumerate(sorted(subgraph_node_ids))}
        idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}

        # Create igraph
        graph = ig.Graph(directed=False)
        graph.add_vertices(len(node_to_idx))

        # Extract edges from cache
        edge_list = []
        edge_weights = []

        with self.read_lock():
            for u in subgraph_node_ids:
                neighbors = owner_cache.get(u, [])
                for v, w in neighbors:
                    if v in node_to_idx and node_to_idx[u] < node_to_idx[v]:
                        # Only add each edge once (undirected graph)
                        edge_list.append((node_to_idx[u], node_to_idx[v]))
                        edge_weights.append(float(w))

        if edge_list:
            graph.add_edges(edge_list)
            graph.es['weight'] = edge_weights

        logger.info(f"Extracted subgraph from cache: {graph.vcount()} nodes, {graph.ecount()} edges")

        return graph, node_to_idx, idx_to_node



    def compute_ppr_push(
        self,
        subgraph_nodes: Set[str],
        reset: Dict[str, float],
        alpha: float = 0.5,
        epsilon: float = 1e-6,
        owner_id: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank using push-based algorithm on cached graph.

        This method is faster than igraph-based PPR for small to medium subgraphs
        as it avoids the overhead of constructing igraph objects.

        Args:
            subgraph_nodes: Set of node IDs in the subgraph
            reset: Reset distribution (dict mapping node_id -> probability, should sum to 1.0)
            alpha: Damping factor (teleport probability)
            epsilon: Convergence threshold

        Returns:
            Dictionary mapping node_id -> PageRank score
        """
        with self.read_lock():
            if not (self._cache_loaded and self._graph_cache):
                logger.warning("Graph cache not loaded, cannot use push-based PPR")
                return {}

        from encapsulation.database.utils.ppr_push import extract_subgraph_adjacency, ppr_push

        with self.read_lock():
            if owner_id is None:
                owner_cache: Dict[str, List[Tuple[str, float]]] = {}
                for shard in self._graph_cache.values():
                    for node_id, neighbors in shard.items():
                        owner_cache.setdefault(node_id, []).extend(neighbors)
            else:
                owner_key = self._owner_key(owner_id)
                owner_cache = self._graph_cache.get(owner_key, {})

            # Extract subgraph adjacency from cache
            subgraph_adj = extract_subgraph_adjacency(owner_cache, subgraph_nodes)

        # Run push-based PPR
        ppr_scores = ppr_push(
            adjacency=subgraph_adj,
            reset=reset,
            alpha=alpha,
            epsilon=epsilon
        )

        return ppr_scores

    def get_cache_version(self) -> int:
        """Get current cache version (incremented on add/delete)."""
        with self.read_lock():
            return self._cache_version
    
    def _invalidate_graph_cache_for_deleted_nodes(self, chunk_ids: List[str], entity_ids: List[str]):
        """Remove deleted nodes and their edges from graph cache."""
        deleted_nodes = set(chunk_ids) | set(entity_ids)
        if not deleted_nodes:
            return

        with self.write_lock():
            if not self._cache_loaded or not self._graph_cache:
                return

            # Remove nodes and edges referencing deleted nodes across owner shards
            for owner_key in list(self._graph_cache.keys()):
                owner_cache = self._graph_cache.get(owner_key, {})

                for node_id in list(owner_cache.keys()):
                    if node_id in deleted_nodes:
                        owner_cache.pop(node_id, None)

                for node_id, neighbors in owner_cache.items():
                    filtered_neighbors = [(n, w) for n, w in neighbors if n not in deleted_nodes]
                    if len(filtered_neighbors) != len(neighbors):
                        owner_cache[node_id] = filtered_neighbors

                if not owner_cache:
                    self._graph_cache.pop(owner_key, None)

            self._entity_chunk_count_cache = self._compute_entity_chunk_count_cache(self._graph_cache)


