import logging
import uuid
from typing import List, Optional, Set, Tuple

import numpy as np

from core.retrieval.graph_retrieveal.similarity_edges import filter_similarity_neighbor_tuples

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

        return [neighbor_id for neighbor_id, _w in neighbors_with_weights]

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
            for neighbor_id, _w in neighbors:
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
            if hasattr(self.graph_store, "get_batch_neighbors_with_weights_and_relations"):
                batch_neighbors = self.graph_store.get_batch_neighbors_with_weights_and_relations(
                    current_layer_list, owner_id=owner_str
                )
            else:
                batch_pairs = self.graph_store.get_batch_neighbors_with_weights(current_layer_list, owner_id=owner_str)
                batch_neighbors = {
                    nid: [(neighbor_id, w, "") for neighbor_id, w in (batch_pairs.get(nid, []) or [])]
                    for nid in current_layer_list
                }

            for node_id in current_layer:
                # Get all neighbors from batch result
                all_neighbors = batch_neighbors.get(node_id, [])
                total_neighbors_before_pruning += len(all_neighbors)

                # Apply pruning
                if not all_neighbors:
                    continue

                # Keep only entity neighbors for expansion (chunks are handled separately).
                all_neighbors = [n for n in all_neighbors if isinstance(n, tuple) and str(n[0] or "").startswith("entity-")]
                # Gate synonymy/similarity edges to reduce multi-hop error amplification.
                all_neighbors = filter_similarity_neighbor_tuples(
                    all_neighbors,
                    hop=hop,
                    relation_name=getattr(self.config, "similarity_edge_relation", "SIMILAR_TO"),
                    max_hops=int(getattr(self.config, "similarity_edge_max_hops", 1)),
                    min_similarity=float(getattr(self.config, "similarity_edge_min_similarity", 0.0)),
                    max_per_node=int(getattr(self.config, "similarity_edge_max_per_node", 0)),
                )

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
                for neighbor_id, _w, _rel in pruned_neighbors:
                    if neighbor_id not in subgraph_nodes:
                        # Only expand to entity nodes
                        if neighbor_id.startswith("entity-"):
                            next_layer.add(neighbor_id)
                            subgraph_nodes.add(neighbor_id)

                # Optionally add chunks connected to new entities (batch query)
            if include_chunks and next_layer:
                next_layer_list = list(next_layer)
                if hasattr(self.graph_store, "get_batch_neighbors_with_weights_and_relations"):
                    entity_batch_neighbors = self.graph_store.get_batch_neighbors_with_weights_and_relations(
                        next_layer_list, owner_id=owner_str
                    )
                else:
                    batch_pairs = self.graph_store.get_batch_neighbors_with_weights(next_layer_list, owner_id=owner_str)
                    entity_batch_neighbors = {
                        nid: [(neighbor_id, w, "") for neighbor_id, w in (batch_pairs.get(nid, []) or [])]
                        for nid in next_layer_list
                    }

                for entity_id in next_layer:
                    entity_neighbors = entity_batch_neighbors.get(entity_id, [])
                    # Prefer chunk neighbors (avoid spending budget on entity-entity edges here).
                    chunk_neighbors = [
                        n for n in entity_neighbors if isinstance(n, tuple) and str(n[0] or "") in chunks_set
                    ]
                    chunk_neighbors.sort(key=lambda x: x[1], reverse=True)

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

                    for en_id, _w, _rel in chunk_neighbors[:max_k]:
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
        *,
        query_doc_scores: Optional[np.ndarray] = None,
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
        phrase_weights = {}
        passage_weights = {}

        # Get entity-to-chunk counts from cache (optimized, no Neo4j query)
        # Collect all entity IDs that appear in facts
        entity_ids_in_facts = set()
        for f in top_k_facts:
            head_id = f[4] if len(f) > 4 else None
            tail_id = f[5] if len(f) > 5 else None
            if head_id and head_id in subgraph_nodes:
                entity_ids_in_facts.add(head_id)
            if tail_id and tail_id in subgraph_nodes:
                entity_ids_in_facts.add(tail_id)

        # Batch get chunk counts from cache
        entity_to_chunk_count = self.graph_store.get_batch_entity_chunk_counts_from_cache(list(entity_ids_in_facts), owner_id=self._owner_to_str(owner_id))

        # Assign weights to entity nodes based on fact scores
        chunk_count_gamma = float(getattr(self.config, "entity_chunk_count_penalty_gamma", 1.0))
        for rank, f in enumerate(top_k_facts):
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            head_id = f[4] if len(f) > 4 else None
            tail_id = f[5] if len(f) > 5 else None
            for entity_id in (head_id, tail_id):
                if not entity_id or entity_id not in subgraph_nodes:
                    continue
                phrase_weights[entity_id] = fact_score
                chunk_count = entity_to_chunk_count.get(entity_id, 0)
                if chunk_count != 0:
                    phrase_weights[entity_id] /= float(chunk_count) ** chunk_count_gamma

        # Assign weights to passage nodes based on dense retrieval
        query_doc_scores = query_doc_scores if query_doc_scores is not None else self._dense_passage_retrieval_scores(query)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids]

        dense_top_file_id: str | None = None
        dense_top_file_ratio = 0.0
        dense_top_file_second_ratio = 0.0
        dense_file_prior_multiplier = 1.0
        dense_file_prior_by_file_id: dict[str, float] = {}
        if bool(getattr(self.config, "dense_file_prior_enabled", False)):
            try:
                top_k = int(getattr(self.config, "dense_file_prior_top_k", 0) or 0)
                min_ratio = float(getattr(self.config, "dense_file_prior_min_ratio", 0.0) or 0.0)
                multiplier = float(getattr(self.config, "dense_file_prior_multiplier", 1.0) or 1.0)
                min_margin = float(getattr(self.config, "dense_file_prior_min_margin", 0.0) or 0.0)
                max_files = int(getattr(self.config, "dense_file_prior_max_files", 1) or 1)
            except Exception:  # noqa: BLE001
                top_k = 0
                min_ratio = 0.0
                multiplier = 1.0
                min_margin = 0.0
                max_files = 1

            source_file_ids = getattr(self, "passage_node_source_file_ids", None)
            if top_k > 0 and isinstance(source_file_ids, list) and source_file_ids:
                from core.retrieval.graph_retrieveal.dense_file_prior import compute_dense_file_prior_multipliers

                ranked_file_ids: list[str | None] = []
                for doc_idx in sorted_doc_ids[:top_k]:
                    i = int(doc_idx)
                    if i < 0 or i >= len(source_file_ids):
                        ranked_file_ids.append(None)
                    else:
                        ranked_file_ids.append(source_file_ids[i])
                dense_file_prior_by_file_id, stats = compute_dense_file_prior_multipliers(
                    ranked_file_ids,
                    top_k=top_k,
                    max_files=max_files,
                    min_ratio=min_ratio,
                    min_margin=min_margin,
                    multiplier=multiplier,
                )
                dense_top_file_id = stats.get("top_file_id")
                dense_top_file_ratio = float(stats.get("top_ratio") or 0.0)
                dense_top_file_second_ratio = float(stats.get("second_ratio") or 0.0)
                if dense_file_prior_by_file_id:
                    dense_file_prior_multiplier = float(multiplier)
                    logger.info(
                        "Dense file prior applied: top_k=%s max_files=%s top_ratio=%.4f second_ratio=%.4f "
                        "min_ratio=%.4f min_margin=%.4f multiplier=%.2f top_file_id=%s prior_files=%s",
                        top_k,
                        max_files,
                        dense_top_file_ratio,
                        dense_top_file_second_ratio,
                        min_ratio,
                        min_margin,
                        dense_file_prior_multiplier,
                        dense_top_file_id,
                        stats.get("prior_files"),
                    )

        # Optional: inject dense top-k chunks into the PPR subgraph before scoring.
        # This is still PPR-based selection, but increases recall for product-name queries when
        # the extracted fact/entity graph does not pull the right file into the expanded subgraph.
        dense_seed_k_raw = getattr(self.config, "dense_seed_subgraph_top_k", 0)
        try:
            dense_seed_k = int(dense_seed_k_raw) if dense_seed_k_raw is not None else 0
        except (TypeError, ValueError):
            dense_seed_k = 0
        dense_seed_k = max(0, dense_seed_k)
        injected_dense_chunk_ids: list[str] = []
        if dense_seed_k > 0:
            for doc_idx in sorted_doc_ids[: max(dense_seed_k * 3, dense_seed_k)]:
                if doc_idx < 0 or int(doc_idx) >= len(self.passage_node_keys):
                    continue
                chunk_id = self.passage_node_keys[int(doc_idx)]
                if not chunk_id or chunk_id in subgraph_nodes:
                    continue
                subgraph_nodes.add(chunk_id)
                injected_dense_chunk_ids.append(chunk_id)
                if len(injected_dense_chunk_ids) >= dense_seed_k:
                    break

            # Keep injected dense chunks connected: also pull in a bounded number of their entity neighbors.
            dense_seed_entity_neighbors_k_raw = getattr(self.config, "dense_seed_subgraph_entity_neighbors_k", 0)
            try:
                dense_seed_entity_neighbors_k = int(dense_seed_entity_neighbors_k_raw or 0)
            except (TypeError, ValueError):
                dense_seed_entity_neighbors_k = 0
            dense_seed_entity_neighbors_k = max(0, dense_seed_entity_neighbors_k)

            injected_dense_entity_ids: set[str] = set()
            if injected_dense_chunk_ids and dense_seed_entity_neighbors_k > 0:
                # Batch-load chunk -> neighbors when supported to avoid N round-trips.
                owner_str = self._owner_to_str(owner_id)
                if hasattr(self.graph_store, "get_batch_neighbors_with_weights_and_relations"):
                    batch_neighbors = self.graph_store.get_batch_neighbors_with_weights_and_relations(
                        list(injected_dense_chunk_ids), owner_id=owner_str
                    )
                    for chunk_id in injected_dense_chunk_ids:
                        neighbors = batch_neighbors.get(chunk_id, []) or []
                        added = 0
                        for neighbor_id, _w, _rel in neighbors:
                            token = str(neighbor_id or "")
                            if not token.startswith("entity-") or token in subgraph_nodes:
                                continue
                            subgraph_nodes.add(token)
                            injected_dense_entity_ids.add(token)
                            added += 1
                            if added >= dense_seed_entity_neighbors_k:
                                break
                else:
                    for chunk_id in injected_dense_chunk_ids:
                        neighbors = self.graph_store.get_neighbors_with_weights(chunk_id, owner_id=owner_str) or []
                        added = 0
                        for neighbor_id, _w in neighbors:
                            token = str(neighbor_id or "")
                            if not token.startswith("entity-") or token in subgraph_nodes:
                                continue
                            subgraph_nodes.add(token)
                            injected_dense_entity_ids.add(token)
                            added += 1
                            if added >= dense_seed_entity_neighbors_k:
                                break

                if injected_dense_entity_ids:
                    logger.info(
                        "Injected dense chunk neighbors into PPR subgraph: dense_seed_entity_neighbors_k=%s injected_entities=%s",
                        dense_seed_entity_neighbors_k,
                        len(injected_dense_entity_ids),
                    )
                tls_getter = getattr(self, "_get_tls", None)
                if callable(tls_getter):
                    try:
                        setattr(tls_getter(), "dense_seed_entity_ids", set(injected_dense_entity_ids))
                    except Exception:  # noqa: BLE001
                        pass
            else:
                tls_getter = getattr(self, "_get_tls", None)
                if callable(tls_getter):
                    try:
                        setattr(tls_getter(), "dense_seed_entity_ids", set())
                    except Exception:  # noqa: BLE001
                        pass
            if injected_dense_chunk_ids:
                logger.info(
                    "Injected dense chunks into PPR subgraph: dense_seed_k=%s injected=%s",
                    dense_seed_k,
                    len(injected_dense_chunk_ids),
                )
            tls_getter = getattr(self, "_get_tls", None)
            if callable(tls_getter):
                try:
                    setattr(tls_getter(), "dense_seed_chunk_ids", set(injected_dense_chunk_ids))
                except Exception:  # noqa: BLE001
                    pass
        else:
            tls_getter = getattr(self, "_get_tls", None)
            if callable(tls_getter):
                try:
                    setattr(tls_getter(), "dense_seed_chunk_ids", set())
                except Exception:  # noqa: BLE001
                    pass

        normalized_dpr_scores = self._min_max_normalize(sorted_doc_scores)

        passage_node_weight = self.config.passage_node_weight
        weighted_scores = normalized_dpr_scores * passage_node_weight

        for doc_id, score in zip(sorted_doc_ids, weighted_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_id = self.passage_node_keys[doc_id]
                if chunk_id in subgraph_nodes:
                    if dense_file_prior_by_file_id and dense_file_prior_multiplier != 1.0:
                        try:
                            fid = self.passage_node_source_file_ids[int(doc_id)]
                        except Exception:  # noqa: BLE001
                            fid = None
                        if fid:
                            mult = dense_file_prior_by_file_id.get(str(fid))
                            if mult:
                                score = float(score) * float(mult)
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
        tls_getter = getattr(self, "_get_tls", None)
        if callable(tls_getter):
            try:
                tls = tls_getter()
                setattr(tls, "dense_top_file_id", dense_top_file_id)
                setattr(tls, "dense_top_file_ratio", dense_top_file_ratio)
                setattr(tls, "dense_top_file_second_ratio", dense_top_file_second_ratio)
                setattr(tls, "dense_file_prior_multiplier", dense_file_prior_multiplier)
                setattr(tls, "dense_file_prior_by_file_id", dense_file_prior_by_file_id)
            except Exception:  # noqa: BLE001
                pass

        # Convert to chunk IDs
        chunk_ids = []
        chunk_scores = []
        for doc_id, score in zip(ppr_sorted_doc_ids, ppr_sorted_doc_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_ids.append(self.passage_node_keys[doc_id])
                chunk_scores.append(score)

        return chunk_ids, chunk_scores, ppr_scores_dict
