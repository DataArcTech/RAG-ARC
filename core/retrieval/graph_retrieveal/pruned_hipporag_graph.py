import logging
import uuid
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple

import numpy as np

from encapsulation.database.utils.pruned_hipporag_utils import compute_entity_id, normalize_entity_text

logger = logging.getLogger(__name__)


class _PrunedHippoRAGGraphMixin:
    def _get_pruned_neighbors_by_weight(self, node_idx: int, entity_relevance_scores: dict = None) -> List[int]:
        """
        Get pruned neighbors for a node using query-aware pruning.

        This method implements query-aware pruning:
        - Neighbors are sorted by edge weight (co-occurrence frequency)
        - The number of neighbors retained (k) is adjusted based on entity relevance to the query
        - More relevant entities get more neighbors, less relevant entities get fewer neighbors

        Args:
            node_idx: Graph index of the node
            entity_relevance_scores: Dict of entity relevance scores for query-aware pruning

        Returns:
            List of neighbor indices (pruned and sorted by weight)
        """
        read_lock = getattr(self.graph_store, "read_lock", None)
        lock_ctx = self.graph_store.read_lock() if callable(read_lock) else nullcontext()
        with lock_ctx:
            graph = self.graph_store.graph
            idx_to_node = self.graph_store.idx_to_node
            node_to_node_stats = self.graph_store.node_to_node_stats

            all_neighbors = graph.neighbors(node_idx, mode="all")

        if not all_neighbors:
            return []

        node_id = idx_to_node.get(node_idx)
        if not node_id:
            return []

        # Collect neighbors with their edge weights
        neighbor_weights = []
        for neighbor_idx in all_neighbors:
            neighbor_id = idx_to_node.get(neighbor_idx)
            if not neighbor_id:
                continue

            # Try both directions (graph is undirected but stats may be stored directionally)
            weight = node_to_node_stats.get((node_id, neighbor_id), 0.0)
            if weight == 0.0:
                weight = node_to_node_stats.get((neighbor_id, node_id), 0.0)

            if weight > 0.0:
                neighbor_weights.append((neighbor_idx, weight))

        if not neighbor_weights:
            return []

        # Sort by weight (descending)
        neighbor_weights.sort(key=lambda x: x[1], reverse=True)

        # Determine max_k (with optional query-aware adjustment)
        base_k = self.config.max_neighbors

        if entity_relevance_scores and node_idx in entity_relevance_scores:
            # Query-aware pruning: adjust k based on entity relevance
            relevance = entity_relevance_scores[node_idx]

            multiplier = self.config.query_aware_multiplier
            min_k = self.config.query_aware_min_k
            max_k_limit = self.config.query_aware_max_k

            max_k = int(base_k * (1 + multiplier * relevance))
            max_k = max(min_k, min(max_k, max_k_limit))

            logger.debug(f"[Query-Aware] Node {node_idx}: relevance={relevance:.3f}, max_k={max_k} (base={base_k})")
        else:
            max_k = base_k

        if len(neighbor_weights) > max_k:
            logger.debug(f"Pruning {len(neighbor_weights)} neighbors to {max_k}")

        # Keep only top-k neighbors
        neighbor_weights = neighbor_weights[:max_k]

        return [idx for idx, _ in neighbor_weights]

    def _expand_subgraph(
        self,
        seed_entity_ids: Set[str],
        entity_relevance_scores: dict = None,
        owner_id: Optional[uuid.UUID] = None,
    ) -> Tuple[Set[int], Set[str]]:
        """
        Expand a subgraph around seed entities using multi-hop traversal.

        The expansion process:
        1. Start with seed entities
        2. Add chunks directly connected to seed entities
        3. For each hop:
           - Expand to neighboring entities (with optional pruning)
           - Optionally add chunks connected to new entities

        Args:
            seed_entity_ids: Set of seed entity IDs to start expansion from
            entity_relevance_scores: Optional relevance scores for query-aware pruning

        Returns:
            Tuple of (subgraph_node_indices, subgraph_chunk_ids)
        """
        read_lock = getattr(self.graph_store, "read_lock", None)
        lock_ctx = self.graph_store.read_lock() if callable(read_lock) else nullcontext()
        with lock_ctx:
            graph = self.graph_store.graph
            subgraph_nodes = set()
            subgraph_chunk_ids = set()

            node_to_idx = self.graph_store.node_to_idx
            idx_to_node = self.graph_store.idx_to_node

            chunks_set = set(self.passage_node_keys)

            # Convert seed entity IDs to graph indices
            seed_entity_indices = set()
            for entity_id in seed_entity_ids:
                if entity_id in node_to_idx:
                    seed_entity_indices.add(node_to_idx[entity_id])

            subgraph_nodes.update(seed_entity_indices)

            # Add chunks directly connected to seed entities
            for entity_idx in seed_entity_indices:
                neighbors = graph.neighbors(entity_idx, mode="all")
                for neighbor_idx in neighbors:
                    neighbor_id = idx_to_node.get(neighbor_idx)
                    if neighbor_id and neighbor_id in chunks_set:
                        subgraph_nodes.add(neighbor_idx)
                        subgraph_chunk_ids.add(neighbor_id)

            logger.info(f"Added {len(subgraph_chunk_ids)} chunks from seed entities")

            include_chunks = self.config.include_chunk_neighbors

            # Multi-hop expansion with query-aware pruning
            current_layer = seed_entity_indices
            for hop in range(self.config.expansion_hops):
                next_layer = set()
                total_neighbors_before_pruning = 0
                total_neighbors_after_pruning = 0

                for node_idx in current_layer:
                    # Get neighbors with query-aware pruning
                    all_neighbors_count = len(graph.neighbors(node_idx, mode="all"))
                    neighbor_indices = self._get_pruned_neighbors_by_weight(
                        node_idx,
                        entity_relevance_scores=entity_relevance_scores,
                    )
                    total_neighbors_before_pruning += all_neighbors_count
                    total_neighbors_after_pruning += len(neighbor_indices)

                    # Process neighbors
                    for neighbor_idx in neighbor_indices:
                        if neighbor_idx not in subgraph_nodes:
                            neighbor_id = idx_to_node.get(neighbor_idx)

                            # Only expand to entity nodes
                            if neighbor_id and neighbor_id.startswith("entity-"):
                                next_layer.add(neighbor_idx)
                                subgraph_nodes.add(neighbor_idx)

                                # Optionally add chunks connected to this entity
                                if include_chunks:
                                    entity_neighbor_indices = self._get_pruned_neighbors_by_weight(
                                        neighbor_idx,
                                        entity_relevance_scores=entity_relevance_scores,
                                    )

                                    for en_idx in entity_neighbor_indices:
                                        en_id = idx_to_node.get(en_idx)
                                        if en_id and en_id in chunks_set:
                                            subgraph_nodes.add(en_idx)
                                            subgraph_chunk_ids.add(en_id)

                logger.info(
                    f"Hop {hop}: {len(current_layer)} nodes, pruned {total_neighbors_before_pruning} → {total_neighbors_after_pruning} neighbors"
                )

                current_layer = next_layer
                if not current_layer:
                    break

            return subgraph_nodes, subgraph_chunk_ids

    def _graph_search_on_subgraph(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[int],
        subgraph_nodes: Set[int],
        owner_id: Optional[uuid.UUID] = None,
    ) -> Tuple[List[str], List[float], np.ndarray]:
        """
        Perform graph search on the expanded subgraph using Personalized PageRank.

        This method:
        1. Assigns weights to entity nodes based on fact scores
        2. Assigns weights to passage nodes based on dense retrieval scores
        3. Runs Personalized PageRank with these weights as reset probabilities
        4. Returns ranked passages based on PPR scores

        Args:
            query: Query string
            query_fact_scores: Scores for retrieved facts
            top_k_facts: Top-k fact triples
            top_k_fact_indices: Indices of top-k facts
            subgraph_nodes: Set of node indices in the subgraph

        Returns:
            Tuple of (chunk_ids, chunk_scores, ppr_scores)
        """
        read_lock = getattr(self.graph_store, "read_lock", None)
        lock_ctx = self.graph_store.read_lock() if callable(read_lock) else nullcontext()
        with lock_ctx:
            num_nodes = len(self.graph_store.graph.vs)
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        # Get entity-to-chunk counts for normalization
        cursor = self.graph_store.conn.cursor()
        owner_str = self._owner_to_str(owner_id)
        if owner_str:
            cursor.execute(
                """
                SELECT cer.entity_id, COUNT(DISTINCT cer.chunk_id) as chunk_count
                FROM chunk_entity_relations cer
                JOIN chunks c ON cer.chunk_id = c.chunk_id
                WHERE c.owner_id = ?
                GROUP BY cer.entity_id
            """,
                (owner_str,),
            )
        else:
            cursor.execute(
                """
                SELECT entity_id, COUNT(DISTINCT chunk_id) as chunk_count
                FROM chunk_entity_relations
                GROUP BY entity_id
            """
            )
        entity_to_chunk_count = {row[0]: row[1] for row in cursor.fetchall()}

        node_to_idx = self.graph_store.node_to_idx

        # Assign weights to entity nodes based on fact scores
        for rank, f in enumerate(top_k_facts):
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for entity_text in [f[0], f[2]]:  # head and tail
                entity_id = compute_entity_id(normalize_entity_text(entity_text), owner_id=owner_str)
                entity_idx = node_to_idx.get(entity_id)

                if entity_idx is not None:
                    phrase_weights[entity_idx] = fact_score

                    # Normalize by chunk count (entities appearing in more chunks get lower weight)
                    chunk_count = entity_to_chunk_count.get(entity_id, 0)
                    if chunk_count != 0:
                        phrase_weights[entity_idx] /= chunk_count

        # Assign weights to passage nodes based on dense retrieval
        query_doc_scores = self._dense_passage_retrieval_scores(query)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids]

        normalized_dpr_scores = self._min_max_normalize(sorted_doc_scores)

        passage_node_weight = self.config.passage_node_weight
        weighted_scores = normalized_dpr_scores * passage_node_weight

        passage_node_ids = np.array([node_to_idx[self.passage_node_keys[doc_id]] for doc_id in sorted_doc_ids], dtype=np.int32)
        passage_weights[passage_node_ids] = weighted_scores

        # Combine entity and passage weights
        node_weights = phrase_weights + passage_weights

        # Zero out weights for nodes outside the subgraph
        subgraph_list = sorted(list(subgraph_nodes))
        all_indices = np.arange(len(node_weights))
        mask = np.isin(all_indices, subgraph_list, invert=True)
        node_weights[mask] = 0.0

        # Fallback to dense retrieval if no weights
        if np.sum(node_weights) == 0:
            logger.warning("No non-zero weights for PPR, falling back to dense retrieval")
            subgraph_chunk_scores = {}
            for i, chunk_id in enumerate(self.passage_node_keys):
                idx = node_to_idx.get(chunk_id)
                if idx is not None and idx in subgraph_nodes:
                    subgraph_chunk_scores[chunk_id] = query_doc_scores[i]
            sorted_items = sorted(subgraph_chunk_scores.items(), key=lambda x: x[1], reverse=True)
            empty_ppr_scores = np.zeros(num_nodes, dtype=np.float32)
            return [item[0] for item in sorted_items], [item[1] for item in sorted_items], empty_ppr_scores

        # Run Personalized PageRank
        ppr_sorted_doc_ids, ppr_sorted_doc_scores, ppr_scores = self._run_ppr_with_weights(
            node_weights=node_weights,
            damping=self.config.damping_factor,
            subgraph_nodes=subgraph_list,
            owner_id=owner_id,
        )

        assert len(ppr_sorted_doc_ids) == len(self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        # Convert to chunk IDs
        chunk_ids = []
        chunk_scores = []
        for doc_id, score in zip(ppr_sorted_doc_ids, ppr_sorted_doc_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_ids.append(self.passage_node_keys[doc_id])
                chunk_scores.append(score)

        return chunk_ids, chunk_scores, ppr_scores

