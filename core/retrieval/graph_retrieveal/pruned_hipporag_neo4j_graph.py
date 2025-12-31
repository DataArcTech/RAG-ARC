import logging
import uuid
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jGraphMixin:
    def _get_pruned_neighbors_by_weight(
        self,
        node_id: str,
        entity_relevance_scores: dict = None,
        owner_id: Optional[uuid.UUID] = None,
    ) -> List[str]:
        """
        Get pruned neighbors for a node using query-aware pruning from Neo4j.

        This method implements query-aware pruning:
        - Neighbors are sorted by edge weight (co-occurrence frequency)
        - The number of neighbors retained (k) is adjusted based on entity relevance to the query
        - More relevant entities get more neighbors, less relevant entities get fewer neighbors

        Args:
            node_id: Node ID (chunk_id or entity_id)
            entity_relevance_scores: Dict of entity relevance scores for query-aware pruning

        Returns:
            List of neighbor IDs (pruned and sorted by weight)
        """
        # Get all neighbors with weights from Neo4j
        neighbors_with_weights = self.graph_store.get_neighbors_with_weights(node_id, owner_id=self._owner_to_str(owner_id))

        if not neighbors_with_weights:
            return []

        # Sort by weight (descending)
        neighbors_with_weights.sort(key=lambda x: x[1], reverse=True)

        # Determine max_k (with optional query-aware adjustment)
        base_k = self.config.max_neighbors

        if entity_relevance_scores and node_id in entity_relevance_scores:
            # Query-aware pruning: adjust k based on entity relevance
            relevance = entity_relevance_scores[node_id]

            multiplier = self.config.query_aware_multiplier
            min_k = self.config.query_aware_min_k
            max_k_limit = self.config.query_aware_max_k

            max_k = int(base_k * (1 + multiplier * relevance))
            max_k = max(min_k, min(max_k, max_k_limit))

            logger.debug(f"[Query-Aware] Node {node_id}: relevance={relevance:.3f}, max_k={max_k} (base={base_k})")
        else:
            max_k = base_k

        if len(neighbors_with_weights) > max_k:
            logger.debug(f"Pruning {len(neighbors_with_weights)} neighbors to {max_k}")

        # Keep only top-k neighbors
        neighbors_with_weights = neighbors_with_weights[:max_k]

        return [neighbor_id for neighbor_id, _ in neighbors_with_weights]

    def _expand_subgraph(
        self,
        seed_entity_ids: Set[str],
        entity_relevance_scores: dict = None,
        owner_id: Optional[uuid.UUID] = None,
    ) -> Tuple[Set[str], Set[str]]:
        """
        Expand a subgraph around seed entities using multi-hop traversal in Neo4j.

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
            Tuple of (subgraph_node_ids, subgraph_chunk_ids)
        """
        subgraph_nodes = set()
        subgraph_chunk_ids = set()

        chunks_set = set(self.passage_node_keys)

        owner_str = self._owner_to_str(owner_id)

        # Start with seed entities
        subgraph_nodes.update(seed_entity_ids)

        # Add chunks directly connected to seed entities
        for entity_id in seed_entity_ids:
            neighbors = self.graph_store.get_neighbors_with_weights(entity_id, owner_id=owner_str)
            for neighbor_id, _ in neighbors:
                if neighbor_id in chunks_set:
                    subgraph_nodes.add(neighbor_id)
                    subgraph_chunk_ids.add(neighbor_id)

        logger.info(f"Added {len(subgraph_chunk_ids)} chunks from seed entities")

        include_chunks = self.config.include_chunk_neighbors

        # Multi-hop expansion with query-aware pruning (optimized with batch queries)
        current_layer = seed_entity_ids
        for hop in range(self.config.expansion_hops):
            next_layer = set()
            total_neighbors_before_pruning = 0
            total_neighbors_after_pruning = 0

            # Batch query for all nodes in current layer
            current_layer_list = list(current_layer)
            batch_neighbors = self.graph_store.get_batch_neighbors_with_weights(current_layer_list, owner_id=owner_str)

            for node_id in current_layer:
                # Get all neighbors from batch result
                all_neighbors = batch_neighbors.get(node_id, [])
                total_neighbors_before_pruning += len(all_neighbors)

                # Apply pruning
                if not all_neighbors:
                    continue

                # Sort by weight and apply query-aware pruning
                all_neighbors.sort(key=lambda x: x[1], reverse=True)

                # Determine max_k
                base_k = self.config.max_neighbors
                if entity_relevance_scores and node_id in entity_relevance_scores:
                    relevance = entity_relevance_scores[node_id]
                    multiplier = self.config.query_aware_multiplier
                    min_k = self.config.query_aware_min_k
                    max_k_limit = self.config.query_aware_max_k
                    max_k = int(base_k * (1 + multiplier * relevance))
                    max_k = max(min_k, min(max_k, max_k_limit))
                else:
                    max_k = base_k

                # Keep top-k neighbors
                pruned_neighbors = all_neighbors[:max_k]
                total_neighbors_after_pruning += len(pruned_neighbors)

                # Process neighbors
                for neighbor_id, _ in pruned_neighbors:
                    if neighbor_id not in subgraph_nodes:
                        # Only expand to entity nodes
                        if neighbor_id.startswith("entity-"):
                            next_layer.add(neighbor_id)
                            subgraph_nodes.add(neighbor_id)

            # Optionally add chunks connected to new entities (batch query)
            if include_chunks and next_layer:
                next_layer_list = list(next_layer)
                entity_batch_neighbors = self.graph_store.get_batch_neighbors_with_weights(next_layer_list, owner_id=owner_str)

                for entity_id in next_layer:
                    entity_neighbors = entity_batch_neighbors.get(entity_id, [])
                    # Sort and prune
                    entity_neighbors.sort(key=lambda x: x[1], reverse=True)

                    base_k = self.config.max_neighbors
                    if entity_relevance_scores and entity_id in entity_relevance_scores:
                        relevance = entity_relevance_scores[entity_id]
                        multiplier = self.config.query_aware_multiplier
                        min_k = self.config.query_aware_min_k
                        max_k_limit = self.config.query_aware_max_k
                        max_k = int(base_k * (1 + multiplier * relevance))
                        max_k = max(min_k, min(max_k, max_k_limit))
                    else:
                        max_k = base_k

                    for en_id, _ in entity_neighbors[:max_k]:
                        if en_id in chunks_set:
                            subgraph_nodes.add(en_id)
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
        subgraph_nodes: Set[str],
        owner_id: Optional[uuid.UUID] = None,
    ) -> Tuple[List[str], List[float], dict]:
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
            subgraph_nodes: Set of node IDs in the subgraph

        Returns:
            Tuple of (chunk_ids, chunk_scores, ppr_scores_dict)
        """
        from encapsulation.database.utils.pruned_hipporag_utils import compute_entity_id, normalize_entity_text

        phrase_weights = {}
        passage_weights = {}

        # Get entity-to-chunk counts from cache (optimized, no Neo4j query)
        # Collect all entity IDs that appear in facts
        owner_str = self._owner_to_str(owner_id)

        entity_ids_in_facts = set()
        for f in top_k_facts:
            for entity_text in [f[0], f[2]]:  # head and tail
                entity_id = compute_entity_id(normalize_entity_text(entity_text), owner_id=owner_str)
                if entity_id in subgraph_nodes:
                    entity_ids_in_facts.add(entity_id)

        # Batch get chunk counts from cache
        entity_to_chunk_count = self.graph_store.get_batch_entity_chunk_counts_from_cache(list(entity_ids_in_facts), owner_id=owner_str)

        # Assign weights to entity nodes based on fact scores
        for rank, f in enumerate(top_k_facts):
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for entity_text in [f[0], f[2]]:  # head and tail
                entity_id = compute_entity_id(normalize_entity_text(entity_text), owner_id=owner_str)

                if entity_id in subgraph_nodes:
                    phrase_weights[entity_id] = fact_score

                    # Normalize by chunk count (entities appearing in more chunks get lower weight)
                    chunk_count = entity_to_chunk_count.get(entity_id, 0)
                    if chunk_count != 0:
                        phrase_weights[entity_id] /= chunk_count

        # Assign weights to passage nodes based on dense retrieval
        query_doc_scores = self._dense_passage_retrieval_scores(query)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids]

        normalized_dpr_scores = self._min_max_normalize(sorted_doc_scores)

        passage_node_weight = self.config.passage_node_weight
        weighted_scores = normalized_dpr_scores * passage_node_weight

        for doc_id, score in zip(sorted_doc_ids, weighted_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_id = self.passage_node_keys[doc_id]
                if chunk_id in subgraph_nodes:
                    passage_weights[chunk_id] = score

        # Combine entity and passage weights
        node_weights = {}
        for node_id in subgraph_nodes:
            node_weights[node_id] = phrase_weights.get(node_id, 0.0) + passage_weights.get(node_id, 0.0)

        # Fallback to dense retrieval if no weights
        if sum(node_weights.values()) == 0:
            logger.warning("No non-zero weights for PPR, falling back to dense retrieval")
            subgraph_chunk_scores = {}
            for i, chunk_id in enumerate(self.passage_node_keys):
                if chunk_id in subgraph_nodes:
                    subgraph_chunk_scores[chunk_id] = query_doc_scores[i]
            sorted_items = sorted(subgraph_chunk_scores.items(), key=lambda x: x[1], reverse=True)
            return [item[0] for item in sorted_items], [item[1] for item in sorted_items], {}

        # Run Personalized PageRank
        ppr_sorted_doc_ids, ppr_sorted_doc_scores, ppr_scores_dict = self._run_ppr_with_weights(
            node_weights=node_weights,
            damping=self.config.damping_factor,
            subgraph_nodes=subgraph_nodes,
            owner_id=owner_id,
        )

        # Convert to chunk IDs
        chunk_ids = []
        chunk_scores = []
        for doc_id, score in zip(ppr_sorted_doc_ids, ppr_sorted_doc_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_ids.append(self.passage_node_keys[doc_id])
                chunk_scores.append(score)

        return chunk_ids, chunk_scores, ppr_scores_dict

