"""
Pruned HippoRAG Retrieval System

Uses the graph store with:
- FAISS Flat for fact retrieval (exact search)
- FAISS HNSW for entity synonymy edges (approximate search)
- numpy arrays for chunk embeddings (brute-force search)

Retrieval approach:
1. Fact retrieval using FAISS Flat (exact search for all fact scores)
2. LLM filtering to extract seed entities
3. Subgraph expansion from seed entities
4. PPR on subgraph with chunk similarity (numpy brute-force)
"""

import logging
import numpy as np
import uuid
from typing import List, Tuple, Set, TYPE_CHECKING, Optional

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_retrieveal.base import BaseGraphRetriever

if TYPE_CHECKING:
    from config.core.retrieval.pruned_hipporag_config import PrunedHippoRAGRetrievalConfig

logger = logging.getLogger(__name__)


class PrunedHippoRAGRetriever(BaseGraphRetriever):
    """
    Pruned HippoRAG Retrieval System
    
    Key differences from original:
    1. Uses FAISS Flat for fact retrieval (exact search)
    2. Uses numpy arrays for chunk embeddings (brute-force search in subgraph)
    3. No HNSW for facts or chunks (only for entity synonymy edges)
    """

    def __init__(self, config: "PrunedHippoRAGRetrievalConfig"):
        """Initialize Pruned HippoRAG retrieval system"""
        super().__init__(config)

        # Build dependencies from config
        self.graph_store = config.graph_config.build()

        # Load existing index if available
        import os
        storage_path = config.graph_config.storage_path
        index_name = config.graph_config.index_name
        if os.path.exists(os.path.join(storage_path, f"{index_name}_graph.pkl")):
            logger.info(f"Loading existing graph index from {storage_path}...")
            self.graph_store.load_index(storage_path, index_name)
        else:
            logger.info(f"No existing index found at {storage_path}, starting with empty graph")

        # Use embedding model from graph_store
        self.embedding_model = self.graph_store.embedding_model

        # Initialize LLM client if configured
        self.llm_client = None
        if config.llm_config is not None:
            try:
                self.llm_client = config.llm_config.build()
                logger.info("LLM client initialized for fact filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Will use fallback filtering.")

        # Build node mappings (without owner_id filter for initialization)
        self._build_node_mappings()

        # Performance optimization: caches
        self._neighbor_cache = {}
        self._all_neighbors_cache = {}

        logger.info("Pruned HippoRAG Retrieval System initialized")
        logger.info(f"  Expansion hops: {config.expansion_hops}")
        logger.info(f"  Include chunk neighbors: {config.include_chunk_neighbors}")
        logger.info(f"  Enable expansion pruning: {config.enable_expansion_pruning}")
        if config.enable_expansion_pruning:
            logger.info(f"    Max neighbors: {config.max_neighbors}")

    def _build_node_mappings(self, owner_id: Optional[uuid.UUID] = None):
        """Build node index mappings (same order as original version)

        Args:
            owner_id: Optional user ID to filter chunks by owner
        """
        self.passage_node_idxs = []
        self.passage_node_keys = []

        # Get chunks from SQLite in insertion order (ORDER BY ROWID)
        # This matches the original version's dict.keys() iteration order
        cursor = self.graph_store.conn.cursor()
        if owner_id:
            cursor.execute("SELECT chunk_id FROM chunks WHERE owner_id = ? ORDER BY ROWID", (str(owner_id),))
        else:
            cursor.execute("SELECT chunk_id FROM chunks ORDER BY ROWID")
        chunk_ids = [row[0] for row in cursor.fetchall()]

        for chunk_id in chunk_ids:
            if chunk_id in self.graph_store.node_to_idx:
                idx = self.graph_store.node_to_idx[chunk_id]
                self.passage_node_idxs.append(idx)
                self.passage_node_keys.append(chunk_id)

        logger.info(f"Built mappings for {len(self.passage_node_idxs)} passage nodes")

    def retrieve(self, query: str, top_k: int = 10, return_subgraph_info: bool = False, owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Main retrieval method using subgraph approach

        Flow:
        1. Get fact scores using FAISS Flat (exact search)
        2. LLM filter to get seed facts and entities
        3. Expand subgraph from seed entities
        4. Add all chunks connected to subgraph entities
        5. Run PPR on subgraph using numpy arrays for chunk similarity

        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            return_subgraph_info: If True, attach subgraph info to first chunk's metadata
            owner_id: Optional user ID to filter chunks by owner

        Returns:
            List of retrieved chunks with scores
        """
        logger.info(f"Retrieving for query: {query} (owner_id={owner_id})")

        # Rebuild node mappings to reflect latest chunks
        # Graph edges are added incrementally during indexing, no need to rebuild
        self._build_node_mappings(owner_id=owner_id)

        # Step 1: Get fact scores using FAISS Flat
        query_fact_scores, fact_ids = self._get_fact_scores_faiss(query)

        if query_fact_scores is None or len(query_fact_scores) == 0:
            logger.warning("No facts found, falling back to dense retrieval")
            return self._dense_passage_retrieval(query, top_k, owner_id=owner_id)

        # Step 2: LLM filter to get seed facts
        if self.config.enable_llm_reranking and self.llm_client:
            top_k_facts, top_k_fact_indices = self._rerank_facts(query, query_fact_scores, fact_ids)
        else:
            # Without LLM, use top-k facts by score
            link_top_k = self.config.fact_retrieval_top_k
            top_k_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
            top_k_facts = self._get_facts_by_indices(top_k_fact_indices, fact_ids)

        if not top_k_facts:
            logger.warning("No facts after reranking, falling back to dense retrieval")
            return self._dense_passage_retrieval(query, top_k, owner_id=owner_id)

        logger.info(f"Selected {len(top_k_facts)} facts after LLM filtering")

        # Step 3: Extract seed entities from facts
        seed_entity_ids = self._extract_entity_ids_from_facts(top_k_facts)

        if not seed_entity_ids:
            logger.warning("No seed entities found, falling back to dense retrieval")
            return self._dense_passage_retrieval(query, top_k, owner_id=owner_id)

        logger.info(f"Extracted {len(seed_entity_ids)} seed entities from {len(top_k_facts)} facts")

        # Step 4: Expand subgraph from seed entities
        subgraph_nodes, subgraph_chunk_ids = self._expand_subgraph(seed_entity_ids)

        logger.info(f"Subgraph: {len(subgraph_nodes)} nodes, {len(subgraph_chunk_ids)} chunks")

        # Step 5: Run PPR on subgraph using numpy arrays
        sorted_doc_ids, sorted_doc_scores, ppr_scores = self._graph_search_on_subgraph(
            query,
            query_fact_scores,
            top_k_facts,
            top_k_fact_indices,
            subgraph_nodes
        )

        # Step 6: Convert to chunks (with owner_id filtering)
        chunks = self._convert_to_chunks(sorted_doc_ids[:top_k], sorted_doc_scores[:top_k], owner_id=owner_id)

        # Step 7: Attach subgraph info if requested
        if return_subgraph_info and chunks:
            # Create node_id to PPR score mapping
            node_to_ppr_score = {}
            idx_to_node = self.graph_store.idx_to_node
            for node_idx in subgraph_nodes:
                node_id = idx_to_node.get(node_idx)
                if node_id and node_idx < len(ppr_scores):
                    node_to_ppr_score[node_id] = float(ppr_scores[node_idx])

            subgraph_info = {
                'subgraph_nodes': list(subgraph_nodes),
                'seed_entity_ids': list(seed_entity_ids),
                'retrieved_chunk_ids': sorted_doc_ids[:top_k],
                'node_ppr_scores': node_to_ppr_score,  # Add PPR scores
                'query': query
            }
            # Attach to first chunk's metadata
            if chunks[0].metadata is None:
                chunks[0].metadata = {}
            chunks[0].metadata['_subgraph_info'] = subgraph_info

        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks

    def _get_fact_scores_faiss(self, query: str) -> Tuple[np.ndarray, List[str]]:
        """
        Get fact scores using FAISS Flat (exact search)

        Returns:
            query_fact_scores: Array of scores for all facts
            fact_ids: List of fact IDs corresponding to scores
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Use FAISS Flat to search all facts (exact search)
        try:
            # Search for all facts (k = total number of facts)
            total_facts = self.graph_store.fact_faiss_db.index.ntotal
            if total_facts == 0:
                logger.warning("No facts in FAISS index")
                return np.array([]), []

            # Prepare query vector
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)

            # Normalize if needed (for cosine similarity)
            if self.graph_store.fact_faiss_db.config.metric == 'cosine' or \
               self.graph_store.fact_faiss_db.config.normalize_L2:
                import faiss
                faiss.normalize_L2(query_vector)

            # Search using FAISS directly
            k = min(total_facts, self.config.fact_retrieval_top_k * 10)
            scores, indices = self.graph_store.fact_faiss_db.index.search(query_vector, k)

            # Flatten results (search returns 2D arrays)
            scores = scores[0]
            indices = indices[0]

            # Get fact IDs from indices
            fact_ids = []
            valid_scores = []
            for idx, score in zip(indices, scores):
                if idx >= 0 and idx in self.graph_store.fact_faiss_db.index_to_docstore_id:
                    fact_id = self.graph_store.fact_faiss_db.index_to_docstore_id[idx]
                    # Skip soft-deleted facts
                    if fact_id not in self.graph_store.fact_faiss_db.deleted_ids:
                        fact_ids.append(fact_id)
                        valid_scores.append(score)

            query_fact_scores = np.array(valid_scores)

            # Min-max normalize
            if len(query_fact_scores) > 0:
                query_fact_scores = self._min_max_normalize(query_fact_scores)

            return query_fact_scores, fact_ids

        except Exception as e:
            logger.error(f"FAISS fact retrieval failed: {e}")
            return np.array([]), []

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding"""
        embedding = self.embedding_model.embed(query)
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        return embedding

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _get_facts_by_indices(self, indices: List[int], fact_ids: List[str]) -> List[Tuple]:
        """
        Get facts by indices from fact_ids list
        
        Args:
            indices: List of indices into fact_ids
            fact_ids: List of fact IDs
            
        Returns:
            List of fact tuples (head, relation, tail)
        """
        facts = []
        cursor = self.graph_store.conn.cursor()

        for idx in indices:
            if idx < len(fact_ids):
                fact_id = fact_ids[idx]
                # Query fact from SQLite
                cursor.execute(
                    "SELECT head, relation, tail FROM facts WHERE fact_id = ?",
                    (fact_id,)
                )
                row = cursor.fetchone()
                if row:
                    facts.append((row[0], row[1], row[2]))

        return facts

    def _extract_entity_ids_from_facts(self, facts: List[Tuple]) -> Set[str]:
        """
        Extract entity IDs from facts

        Args:
            facts: List of fact tuples (head_name, relation, tail_name)

        Returns:
            Set of entity IDs
        """
        entity_ids = set()
        cursor = self.graph_store.conn.cursor()

        # Build entity name to ID mapping
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entity_name_to_id = {name: eid for eid, name in cursor.fetchall()}

        # Convert entity names to IDs
        for head_name, _, tail_name in facts:
            head_id = entity_name_to_id.get(head_name)
            tail_id = entity_name_to_id.get(tail_name)
            if head_id:
                entity_ids.add(head_id)
            if tail_id:
                entity_ids.add(tail_id)

        return entity_ids

    def _get_pruned_neighbors_by_weight(self, node_idx: int) -> List[int]:
        """
        Get pruned neighbors for a node by sorting by edge weight

        Args:
            node_idx: Node index

        Returns:
            List of neighbor indices sorted by edge weight (descending), limited by max_neighbors
        """
        graph = self.graph_store.graph

        # Get all neighbors
        all_neighbors = graph.neighbors(node_idx, mode="all")

        if not all_neighbors:
            return []

        # Get neighbor weights
        neighbor_weights = []
        for neighbor_idx in all_neighbors:
            # Try to get edge in both directions
            edge_id = graph.get_eid(node_idx, neighbor_idx, error=False)
            if edge_id == -1:
                edge_id = graph.get_eid(neighbor_idx, node_idx, error=False)

            if edge_id != -1:
                weight = graph.es[edge_id]['weight']
                neighbor_weights.append((neighbor_idx, weight))

        if not neighbor_weights:
            return []

        # Sort by weight (descending)
        neighbor_weights.sort(key=lambda x: x[1], reverse=True)

        # Prune to max_neighbors
        max_k = self.config.max_neighbors

        # Log pruning info (first time only)
        if len(neighbor_weights) > max_k:
            logger.debug(f"Pruning {len(neighbor_weights)} neighbors to {max_k}")

        neighbor_weights = neighbor_weights[:max_k]

        # Return neighbor indices
        return [idx for idx, _ in neighbor_weights]

    def _expand_subgraph(
        self,
        seed_entity_ids: Set[str]
    ) -> Tuple[Set[int], Set[str]]:
        """
        Expand subgraph from seed entities

        Args:
            seed_entity_ids: Set of seed entity IDs

        Returns:
            subgraph_nodes: Set of node indices in subgraph
            subgraph_chunk_ids: Set of chunk IDs in subgraph
        """
        graph = self.graph_store.graph
        subgraph_nodes = set()
        subgraph_chunk_ids = set()

        node_to_idx = self.graph_store.node_to_idx
        idx_to_node = self.graph_store.idx_to_node

        # Get all chunk IDs (for checking if a node is a chunk)
        chunks_set = set(self.passage_node_keys)

        # Convert seed entity IDs to vertex indices
        seed_entity_indices = set()
        for entity_id in seed_entity_ids:
            if entity_id in node_to_idx:
                seed_entity_indices.add(node_to_idx[entity_id])

        # Start with seed entities
        subgraph_nodes.update(seed_entity_indices)

        # First, add all chunks connected to seed entities (same as original)
        for entity_idx in seed_entity_indices:
            neighbors = graph.neighbors(entity_idx, mode="all")
            for neighbor_idx in neighbors:
                neighbor_id = idx_to_node.get(neighbor_idx)
                if neighbor_id and neighbor_id in chunks_set:
                    subgraph_nodes.add(neighbor_idx)
                    subgraph_chunk_ids.add(neighbor_id)

        logger.info(f"Added {len(subgraph_chunk_ids)} chunks from seed entities")

        # Then, expand k-hops from seed entities
        enable_pruning = self.config.enable_expansion_pruning
        include_chunks = getattr(self.config, 'include_chunk_neighbors', True)

        current_layer = seed_entity_indices
        for hop in range(self.config.expansion_hops):
            next_layer = set()
            total_neighbors_before_pruning = 0
            total_neighbors_after_pruning = 0

            for node_idx in current_layer:
                # Get neighbors with pruning
                if enable_pruning:
                    all_neighbors_count = len(graph.neighbors(node_idx, mode="all"))
                    neighbor_indices = self._get_pruned_neighbors_by_weight(node_idx)
                    total_neighbors_before_pruning += all_neighbors_count
                    total_neighbors_after_pruning += len(neighbor_indices)
                else:
                    neighbor_indices = graph.neighbors(node_idx, mode="all")

                for neighbor_idx in neighbor_indices:
                    if neighbor_idx not in subgraph_nodes:
                        neighbor_id = idx_to_node.get(neighbor_idx)

                        # Add entity neighbors
                        if neighbor_id and neighbor_id.startswith("entity-"):
                            next_layer.add(neighbor_idx)
                            subgraph_nodes.add(neighbor_idx)

                            # Add chunks connected to this entity (if include_chunks=True)
                            if include_chunks:
                                # Get entity's neighbors with pruning
                                if enable_pruning:
                                    entity_neighbor_indices = self._get_pruned_neighbors_by_weight(neighbor_idx)
                                else:
                                    entity_neighbor_indices = graph.neighbors(neighbor_idx, mode="all")

                                for en_idx in entity_neighbor_indices:
                                    en_id = idx_to_node.get(en_idx)
                                    if en_id and en_id in chunks_set:
                                        subgraph_nodes.add(en_idx)
                                        subgraph_chunk_ids.add(en_id)

            logger.info(f"Hop {hop}: {len(current_layer)} nodes, pruned {total_neighbors_before_pruning} → {total_neighbors_after_pruning} neighbors")

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
        subgraph_nodes: Set[int]
    ) -> Tuple[List[str], List[float], np.ndarray]:
        """
        Run PPR on subgraph following HippoRAG's approach

        This follows the original HippoRAG algorithm:
        1. Compute phrase weights from fact scores (weighted by entity chunk count)
        2. Compute passage weights from dense retrieval scores
        3. Combine into node_weights for PPR personalization
        4. Run PPR with combined weights
        5. Return sorted passage nodes by PPR scores

        Args:
            query: Query string
            query_fact_scores: Fact scores array
            top_k_facts: Top-k facts after reranking
            top_k_fact_indices: Indices of top-k facts
            subgraph_nodes: Set of node indices in subgraph
            fact_ids: List of fact IDs

        Returns:
            sorted_doc_ids: List of chunk IDs sorted by score
            sorted_doc_scores: List of scores
            ppr_scores: Array of PPR scores for all nodes
        """
        import re

        num_nodes = len(self.graph_store.graph.vs)
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        # Get entity to chunk count from SQLite
        cursor = self.graph_store.conn.cursor()
        cursor.execute('''
            SELECT entity_id, COUNT(DISTINCT chunk_id) as chunk_count
            FROM chunk_entity_relations
            GROUP BY entity_id
        ''')
        entity_to_chunk_count = {row[0]: row[1] for row in cursor.fetchall()}

        node_to_idx = self.graph_store.node_to_idx
        pattern = re.compile('[^A-Za-z0-9 ]')

        # Step 1: Compute phrase weights from fact scores
        for rank, f in enumerate(top_k_facts):
            subject_phrase = pattern.sub(' ', f[0].lower()).strip()
            object_phrase = pattern.sub(' ', f[2].lower()).strip()
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                # Convert phrase to entity_id
                phrase_key = self._compute_entity_id(phrase)
                phrase_idx = node_to_idx.get(phrase_key, None)

                if phrase_idx is not None:
                    phrase_weights[phrase_idx] = fact_score

                    # Weight by inverse chunk count (entities in fewer chunks get higher weight)
                    chunk_count = entity_to_chunk_count.get(phrase_key, 0)
                    if chunk_count != 0:
                        phrase_weights[phrase_idx] /= chunk_count

        # Step 2: Compute passage weights from dense retrieval
        # Get sorted doc IDs and scores (same as original version)
        query_doc_scores = self._dense_passage_retrieval_scores(query)

        # Sort by score (descending)
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids]

        # Normalize sorted scores
        normalized_dpr_scores = self._min_max_normalize(sorted_doc_scores)

        passage_node_weight = self.config.passage_node_weight
        weighted_scores = normalized_dpr_scores * passage_node_weight

        # Add passage weights for ALL chunks (using sorted order)
        passage_node_ids = np.array([node_to_idx[self.passage_node_keys[doc_id]]
                                     for doc_id in sorted_doc_ids], dtype=np.int32)
        passage_weights[passage_node_ids] = weighted_scores

        # Step 3: Combine phrase and passage weights
        node_weights = phrase_weights + passage_weights

        # Step 4: Zero out weights for nodes NOT in subgraph
        subgraph_list = sorted(list(subgraph_nodes))
        all_indices = np.arange(len(node_weights))
        mask = np.isin(all_indices, subgraph_list, invert=True)
        node_weights[mask] = 0.0

        # Ensure we have non-zero weights
        if np.sum(node_weights) == 0:
            logger.warning("No non-zero weights for PPR, falling back to dense retrieval")
            # Return dense retrieval results for subgraph chunks only
            subgraph_chunk_scores = {}
            for i, chunk_id in enumerate(self.passage_node_keys):
                idx = node_to_idx.get(chunk_id)
                if idx is not None and idx in subgraph_nodes:
                    subgraph_chunk_scores[chunk_id] = query_doc_scores[i]
            sorted_items = sorted(subgraph_chunk_scores.items(), key=lambda x: x[1], reverse=True)
            # Return empty PPR scores array
            empty_ppr_scores = np.zeros(num_nodes, dtype=np.float32)
            return [item[0] for item in sorted_items], [item[1] for item in sorted_items], empty_ppr_scores

        # Step 5: Run PPR with combined weights
        ppr_sorted_doc_ids, ppr_sorted_doc_scores, ppr_scores = self._run_ppr_with_weights(
            node_weights=node_weights,
            damping=self.config.damping_factor,
            subgraph_nodes=subgraph_list
        )

        # Verify result length (same as original version)
        assert len(ppr_sorted_doc_ids) == len(self.passage_node_idxs), \
            f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        # Convert doc_ids (indices) to chunk_ids (strings)
        chunk_ids = []
        chunk_scores = []
        for doc_id, score in zip(ppr_sorted_doc_ids, ppr_sorted_doc_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_ids.append(self.passage_node_keys[doc_id])
                chunk_scores.append(score)

        return chunk_ids, chunk_scores, ppr_scores

    def _compute_entity_id(self, entity_name: str) -> str:
        """Compute entity ID from entity name (same as graph store)"""
        import hashlib
        return "entity-" + hashlib.md5(entity_name.encode()).hexdigest()

    def _run_ppr_with_weights(
        self,
        node_weights: np.ndarray,
        damping: float = 0.5,
        subgraph_nodes: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Personalized PageRank with node weights as reset probabilities

        This follows the original HippoRAG approach where node_weights contains
        combined phrase and passage weights.

        Args:
            node_weights: Array of weights for all nodes (phrase + passage weights)
            damping: Damping factor
            subgraph_nodes: List of node indices in subgraph (sorted)

        Returns:
            sorted_doc_ids: Array of document indices sorted by PPR score
            sorted_doc_scores: Array of PPR scores
            pagerank_scores: Array of PPR scores for all nodes
        """
        # Handle NaN and negative values
        node_weights = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)

        # Decide whether to use induced subgraph (only if < 30% of full graph)
        use_subgraph = (subgraph_nodes is not None and
                       len(subgraph_nodes) > 0 and
                       len(subgraph_nodes) < len(self.graph_store.node_to_idx) * 0.3)

        if use_subgraph:
            logger.info(f"Using induced subgraph PPR: {len(subgraph_nodes)}/{len(self.graph_store.node_to_idx)} nodes")

            # Create induced subgraph
            subgraph = self.graph_store.graph.induced_subgraph(subgraph_nodes)

            # Map reset probabilities to subgraph
            subgraph_reset = [node_weights[node_idx] for node_idx in subgraph_nodes]

            # Normalize reset probabilities
            reset_sum = sum(subgraph_reset)
            if reset_sum > 0:
                subgraph_reset = [r / reset_sum for r in subgraph_reset]
            else:
                logger.warning("All reset probabilities are zero")
                return {}

            # Run PPR on subgraph
            try:
                subgraph_pagerank = subgraph.personalized_pagerank(
                    damping=damping,
                    directed=False,
                    weights='weight',
                    reset=subgraph_reset,
                    implementation='prpack'
                )

                # Map back to full graph indices
                pagerank_scores = [0.0] * len(self.graph_store.node_to_idx)
                for i, node_idx in enumerate(subgraph_nodes):
                    pagerank_scores[node_idx] = subgraph_pagerank[i]

            except Exception as e:
                logger.error(f"Subgraph PPR failed: {e}")
                # Return empty results
                return np.array([]), np.array([]), np.array([])
        else:
            # Run PPR on full graph
            logger.info(f"Using full graph PPR: {len(self.graph_store.node_to_idx)} nodes")

            # Normalize reset probabilities
            reset_sum = node_weights.sum()
            if reset_sum > 0:
                reset_prob = node_weights / reset_sum
            else:
                logger.warning("All reset probabilities are zero")
                return np.array([]), np.array([]), np.array([])

            try:
                pagerank_scores = self.graph_store.graph.personalized_pagerank(
                    damping=damping,
                    directed=False,
                    weights='weight',
                    reset=reset_prob.tolist(),
                    implementation='prpack'
                )

            except Exception as e:
                logger.error(f"Full graph PPR failed: {e}")
                return np.array([]), np.array([]), np.array([])

        # Extract document scores (same as original version)
        pagerank_array = np.array(pagerank_scores)
        doc_scores = pagerank_array[self.passage_node_idxs]

        # Sort by score
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids]

        return sorted_doc_ids, sorted_doc_scores, pagerank_array

    def _dense_passage_retrieval_scores(self, query: str) -> np.ndarray:
        """
        Get dense passage retrieval scores using numpy arrays (brute-force)

        Args:
            query: Query string

        Returns:
            Array of scores for all passages (NOT normalized - caller will normalize)
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Build passage embeddings array in the same order as passage_node_keys
        passage_embeddings_list = []
        for chunk_id in self.passage_node_keys:
            if chunk_id in self.graph_store.chunk_embeddings:
                passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
            else:
                # Use zero vector for missing embeddings
                if passage_embeddings_list:
                    embedding_dim = len(passage_embeddings_list[0])
                else:
                    embedding_dim = 1024  # Default dimension
                passage_embeddings_list.append(np.zeros(embedding_dim))

        if not passage_embeddings_list:
            logger.warning("No passage embeddings available")
            return np.zeros(len(self.passage_node_keys))

        passage_embeddings_array = np.array(passage_embeddings_list)

        # Compute similarity (dot product) - vectorized operation
        query_doc_scores = np.dot(passage_embeddings_array, query_embedding)

        # Do NOT normalize here - caller will normalize (same as original version)
        return query_doc_scores

    def _dense_passage_retrieval(self, query: str, top_k: int = 10, owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Fallback: Dense passage retrieval using numpy arrays

        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            owner_id: Optional user ID to filter chunks by owner

        Returns:
            List of retrieved chunks
        """
        query_doc_scores = self._dense_passage_retrieval_scores(query)

        # Get top-k indices
        top_k_indices = np.argsort(query_doc_scores)[-top_k:][::-1]

        # Get chunk IDs and scores
        chunk_ids = [self.passage_node_keys[i] for i in top_k_indices if i < len(self.passage_node_keys)]
        scores = [query_doc_scores[i] for i in top_k_indices if i < len(query_doc_scores)]

        # Convert to chunks (with owner_id filtering)
        chunks = self._convert_to_chunks(chunk_ids, scores, owner_id=owner_id)

        return chunks

    def _convert_to_chunks(self, chunk_ids: List[str], scores: List[float], owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Convert chunk IDs and scores to Chunk objects

        Args:
            chunk_ids: List of chunk IDs
            scores: List of scores
            owner_id: Optional user ID to filter chunks by owner

        Returns:
            List of Chunk objects with scores in metadata (filtered by owner_id if provided)
        """
        chunks = []
        cursor = self.graph_store.conn.cursor()

        for chunk_id, score in zip(chunk_ids, scores):
            # Query chunk from SQLite with optional owner_id filter
            if owner_id is not None:
                cursor.execute(
                    "SELECT content, owner_id, metadata FROM chunks WHERE chunk_id = ? AND owner_id = ?",
                    (chunk_id, str(owner_id))
                )
            else:
                cursor.execute(
                    "SELECT content, owner_id, metadata FROM chunks WHERE chunk_id = ?",
                    (chunk_id,)
                )

            row = cursor.fetchone()
            if row:
                import json
                content = row[0]
                chunk_owner_id = row[1]
                metadata = json.loads(row[2]) if row[2] else {}
                metadata['score'] = float(score)

                chunk = Chunk(
                    id=chunk_id,
                    content=content,
                    owner_id=chunk_owner_id,
                    metadata=metadata
                )
                chunks.append(chunk)

        return chunks

    def _rerank_facts(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        fact_ids: List[str]
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Rerank facts using LLM

        Args:
            query: Query string
            query_fact_scores: Fact scores array
            fact_ids: List of fact IDs

        Returns:
            top_k_facts: List of fact tuples (head, relation, tail)
            top_k_fact_indices: List of fact indices after reranking
        """
        link_top_k = self.config.fact_retrieval_top_k

        # Get top candidate facts by score
        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
        candidate_facts = self._get_facts_by_indices(candidate_fact_indices, fact_ids)

        # Call LLM rerank filter
        try:
            top_k_facts, top_k_fact_indices = self._llm_rerank_filter(
                query,
                candidate_facts,
                candidate_fact_indices,
                len_after_rerank=self.config.max_facts_after_reranking
            )
            logger.info(f"LLM reranked {len(candidate_facts)} facts to {len(top_k_facts)}")
            return top_k_facts, top_k_fact_indices
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}, using top facts by score")
            # Fallback: use top facts by score
            max_facts = min(self.config.max_facts_after_reranking, len(candidate_facts))
            return candidate_facts[:max_facts], candidate_fact_indices[:max_facts]

    def _llm_rerank_filter(
        self,
        query: str,
        candidate_facts: List[Tuple],
        candidate_fact_indices: List[int],
        len_after_rerank: int = 5
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Use LLM to filter and rerank facts

        Args:
            query: Query string
            candidate_facts: List of candidate fact tuples
            candidate_fact_indices: Indices of candidate facts
            len_after_rerank: Number of facts to keep after reranking

        Returns:
            top_k_facts: List of top-k fact tuples
            top_k_fact_indices: List of top-k fact indices
        """
        # Format facts for LLM
        facts_text = "\n".join([
            f"{i+1}. {head} - {relation} - {tail}"
            for i, (head, relation, tail) in enumerate(candidate_facts)
        ])

        # Create prompt
        prompt = f"""Given the query: "{query}"

Select the {len_after_rerank} most relevant facts from the following list:

{facts_text}

Return only the numbers of the selected facts, separated by commas (e.g., "1,3,5").
"""

        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)

        # Parse response
        try:
            selected_indices = [int(x.strip()) - 1 for x in response.split(",")]
            selected_indices = [i for i in selected_indices if 0 <= i < len(candidate_facts)]

            if not selected_indices:
                raise ValueError("No valid indices in LLM response")

            # Get selected facts
            top_k_facts = [candidate_facts[i] for i in selected_indices[:len_after_rerank]]
            top_k_fact_indices = [candidate_fact_indices[i] for i in selected_indices[:len_after_rerank]]

            return top_k_facts, top_k_fact_indices

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback: use top facts by score
            max_facts = min(len_after_rerank, len(candidate_facts))
            return candidate_facts[:max_facts], candidate_fact_indices[:max_facts]

