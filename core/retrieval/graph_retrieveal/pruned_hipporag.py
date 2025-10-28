import logging
import numpy as np
import uuid
import os
import json
from typing import List, Tuple, Set, TYPE_CHECKING, Optional
from collections import defaultdict

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_retrieveal.base import BaseGraphRetriever
from encapsulation.database.utils.pruned_hipporag_utils import normalize_entity_text, compute_entity_id

if TYPE_CHECKING:
    from config.core.retrieval.pruned_hipporag_config import PrunedHippoRAGRetrievalConfig

logger = logging.getLogger(__name__)


class PrunedHippoRAGRetriever(BaseGraphRetriever):
    """
    Pruned HippoRAG Retrieval System.

    This retriever implements a graph-based retrieval approach that:
    1. Retrieves relevant facts from a knowledge graph using dense retrieval (FAISS)
    2. Optionally reranks facts using an LLM to improve relevance
    3. Extracts seed entities from top-ranked facts
    4. Expands a subgraph around seed entities with optional pruning
    5. Performs Personalized PageRank (PPR) on the subgraph to rank passages

    The "pruned" aspect refers to limiting the number of neighbors during graph expansion
    to balance retrieval quality and computational efficiency.
    """

    def __init__(self, config: "PrunedHippoRAGRetrievalConfig"):
        """
        Initialize the Pruned HippoRAG Retriever.

        Args:
            config: Configuration object containing all retrieval parameters
        """
        super().__init__(config)

        # Build and load graph store
        self.graph_store = config.graph_config.build()

        storage_path = config.graph_config.storage_path
        index_name = config.graph_config.index_name
        if os.path.exists(os.path.join(storage_path, f"{index_name}_graph.pkl")):
            logger.info(f"Loading existing graph index from {storage_path}...")
            self.graph_store.load_index(storage_path, index_name)
        else:
            logger.info(f"No existing index found at {storage_path}, starting with empty graph")

        self.embedding_model = self.graph_store.embedding_model

        # Initialize optional LLM client for fact reranking
        self.llm_client = None
        if config.llm_config is not None:
            try:
                self.llm_client = config.llm_config.build()
                logger.info("LLM client initialized for fact filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Will use fallback filtering.")

        # Build initial node mappings
        self._build_node_mappings()

        logger.info("Pruned HippoRAG Retrieval System initialized")
        logger.info(f"  Expansion hops: {config.expansion_hops}")
        logger.info(f"  Include chunk neighbors: {config.include_chunk_neighbors}")
        logger.info(f"  Enable query-aware pruning: {config.enable_pruning}")
        if config.enable_pruning:
            logger.info(f"    Base max neighbors: {config.max_neighbors}")
            logger.info(f"    Query-aware multiplier: {config.query_aware_multiplier}")
            logger.info(f"    Min/Max neighbors: {config.query_aware_min_k}/{config.query_aware_max_k}")

    def _build_node_mappings(self, owner_id: Optional[uuid.UUID] = None):
        """
        Build mappings between passage nodes and their indices in the graph.

        This creates two parallel lists:
        - passage_node_idxs: Graph indices for passage/chunk nodes
        - passage_node_keys: Chunk IDs corresponding to those indices

        Args:
            owner_id: Optional owner ID to filter chunks by ownership
        """
        self.passage_node_idxs = []
        self.passage_node_keys = []

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
        Main retrieval method implementing the Pruned HippoRAG algorithm.

        Retrieval pipeline:
        1. Retrieve relevant facts using FAISS dense retrieval
        2. Optionally rerank facts using LLM
        3. Extract seed entities from top facts
        4. Expand subgraph around seed entities
        5. Perform Personalized PageRank on subgraph
        6. Return top-k ranked chunks

        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            return_subgraph_info: Whether to include subgraph metadata in results
            owner_id: Optional owner ID to filter chunks by ownership

        Returns:
            List of retrieved Chunk objects, ranked by relevance
        """
        logger.info(f"Retrieving for query: {query} (owner_id={owner_id})")

        # Rebuild node mappings for the current owner
        self._build_node_mappings(owner_id=owner_id)

        # Step 1: Retrieve relevant facts
        query_fact_scores, fact_ids = self._get_fact_scores_faiss(query)

        if query_fact_scores is None or len(query_fact_scores) == 0:
            logger.warning("No facts found, falling back to dense retrieval")
            return self._dense_passage_retrieval(query, top_k, owner_id=owner_id)

        # Step 2: Rerank facts (optional)
        if self.config.enable_llm_reranking and self.llm_client:
            top_k_facts, top_k_fact_indices = self._rerank_facts(query, query_fact_scores, fact_ids)
        else:
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

        # Step 4: Compute entity relevance scores for query-aware pruning
        entity_relevance_scores = None
        if self.config.enable_pruning:
            entity_relevance_scores = self._compute_entity_relevance_scores(
                seed_entity_ids,
                top_k_facts,
                query_fact_scores,
                top_k_fact_indices
            )
            logger.info(f"[Query-Aware] Computed relevance scores for {len(entity_relevance_scores)} entities")

        # Step 5: Expand subgraph around seed entities
        subgraph_nodes, subgraph_chunk_ids = self._expand_subgraph(
            seed_entity_ids,
            entity_relevance_scores=entity_relevance_scores
        )

        logger.info(f"Subgraph: {len(subgraph_nodes)} nodes, {len(subgraph_chunk_ids)} chunks")

        # Step 6: Perform graph search using Personalized PageRank
        sorted_doc_ids, sorted_doc_scores, ppr_scores = self._graph_search_on_subgraph(
            query,
            query_fact_scores,
            top_k_facts,
            top_k_fact_indices,
            subgraph_nodes
        )

        # Step 7: Convert to Chunk objects
        chunks = self._convert_to_chunks(sorted_doc_ids[:top_k], sorted_doc_scores[:top_k], owner_id=owner_id)

        # Optionally attach subgraph information for visualization
        if return_subgraph_info and chunks:
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
                'node_ppr_scores': node_to_ppr_score,
                'query': query
            }
            if chunks[0].metadata is None:
                chunks[0].metadata = {}
            chunks[0].metadata['_subgraph_info'] = subgraph_info

        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks

    def _get_fact_scores_faiss(self, query: str) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve relevant facts using FAISS dense retrieval.

        Args:
            query: Query string

        Returns:
            Tuple of (normalized_scores, fact_ids)
        """
        query_embedding = self._get_query_embedding(query)

        try:
            # Check if fact index exists and is initialized
            if self.graph_store.fact_faiss_db.index is None:
                logger.warning("Fact FAISS index is not initialized")
                return np.array([]), []

            total_facts = self.graph_store.fact_faiss_db.index.ntotal
            if total_facts == 0:
                logger.warning("No facts in FAISS index")
                return np.array([]), []

            query_vector = query_embedding.reshape(1, -1).astype(np.float32)

            # Normalize query vector for cosine similarity
            if self.graph_store.fact_faiss_db.config.metric == 'cosine' or \
               self.graph_store.fact_faiss_db.config.normalize_L2:
                import faiss
                faiss.normalize_L2(query_vector)

            # Retrieve top-k facts (with buffer for filtering)
            k = min(total_facts, self.config.fact_retrieval_top_k * 10)
            scores, indices = self.graph_store.fact_faiss_db.index.search(query_vector, k)

            scores = scores[0]
            indices = indices[0]

            # Filter out deleted facts
            fact_ids = []
            valid_scores = []
            for idx, score in zip(indices, scores):
                if idx >= 0 and idx in self.graph_store.fact_faiss_db.index_to_docstore_id:
                    fact_id = self.graph_store.fact_faiss_db.index_to_docstore_id[idx]
                    if fact_id not in self.graph_store.fact_faiss_db.deleted_ids:
                        fact_ids.append(fact_id)
                        valid_scores.append(score)

            query_fact_scores = np.array(valid_scores)

            # Normalize scores to [0, 1] range
            if len(query_fact_scores) > 0:
                query_fact_scores = self._min_max_normalize(query_fact_scores)

            return query_fact_scores, fact_ids

        except Exception as e:
            logger.error(f"FAISS fact retrieval failed: {e}")
            return np.array([]), []

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query string."""
        embedding = self.embedding_model.embed(query)
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        return embedding

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _get_facts_by_indices(self, indices: List[int], fact_ids: List[str]) -> List[Tuple]:
        """
        Retrieve fact triples from database by their indices.

        Args:
            indices: List of indices into fact_ids
            fact_ids: List of fact IDs

        Returns:
            List of fact triples (head, relation, tail)
        """
        facts = []
        cursor = self.graph_store.conn.cursor()

        for idx in indices:
            if idx < len(fact_ids):
                fact_id = fact_ids[idx]
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
        Extract unique entity IDs from fact triples.

        Args:
            facts: List of fact triples (head, relation, tail)

        Returns:
            Set of entity IDs appearing in the facts
        """
        entity_ids = set()
        cursor = self.graph_store.conn.cursor()

        # Build entity name to ID mapping
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entity_name_to_id = {name: eid for eid, name in cursor.fetchall()}

        # Extract entities from fact heads and tails
        for head_name, _, tail_name in facts:
            head_id = entity_name_to_id.get(head_name)
            tail_id = entity_name_to_id.get(tail_name)
            if head_id:
                entity_ids.add(head_id)
            if tail_id:
                entity_ids.add(tail_id)

        return entity_ids

    def _compute_entity_relevance_scores(
        self,
        seed_entity_ids: Set[str],
        top_k_facts: List[Tuple],
        query_fact_scores: np.ndarray,
        top_k_fact_indices: List[int]
    ) -> dict:
        """
        Compute relevance scores for entities based on their associated fact scores.

        This is used for query-aware pruning: entities with higher relevance scores
        will have more neighbors retained during graph expansion.

        Args:
            seed_entity_ids: Set of seed entity IDs
            top_k_facts: List of top-k fact triples
            query_fact_scores: Scores for all retrieved facts
            top_k_fact_indices: Indices of top-k facts

        Returns:
            Dictionary mapping entity graph indices to relevance scores
        """
        # Collect fact scores for each entity
        entity_to_fact_scores = defaultdict(list)

        for fact_idx, fact in zip(top_k_fact_indices, top_k_facts):
            head_id = compute_entity_id(normalize_entity_text(fact[0]))
            tail_id = compute_entity_id(normalize_entity_text(fact[2]))

            fact_score = float(query_fact_scores[fact_idx]) if query_fact_scores.ndim > 0 else float(query_fact_scores)

            entity_to_fact_scores[head_id].append(fact_score)
            entity_to_fact_scores[tail_id].append(fact_score)

        # Compute average relevance score for each entity
        entity_relevance_scores = {}
        node_to_idx = self.graph_store.node_to_idx

        for entity_id in seed_entity_ids:
            if entity_id in entity_to_fact_scores:
                avg_score = float(np.mean(entity_to_fact_scores[entity_id]))

                if entity_id in node_to_idx:
                    entity_idx = node_to_idx[entity_id]
                    entity_relevance_scores[entity_idx] = avg_score

        if entity_relevance_scores:
            logger.info(f"[Query-Aware] Entity relevance scores: min={min(entity_relevance_scores.values()):.3f}, "
                       f"max={max(entity_relevance_scores.values()):.3f}, "
                       f"avg={np.mean(list(entity_relevance_scores.values())):.3f}")

        return entity_relevance_scores

    def _get_pruned_neighbors_by_weight(
        self,
        node_idx: int,
        entity_relevance_scores: dict = None
    ) -> List[int]:
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
        entity_relevance_scores: dict = None
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
                    entity_relevance_scores=entity_relevance_scores
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
                                    entity_relevance_scores=entity_relevance_scores
                                )

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
        num_nodes = len(self.graph_store.graph.vs)
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        # Get entity-to-chunk counts for normalization
        cursor = self.graph_store.conn.cursor()
        cursor.execute('''
            SELECT entity_id, COUNT(DISTINCT chunk_id) as chunk_count
            FROM chunk_entity_relations
            GROUP BY entity_id
        ''')
        entity_to_chunk_count = {row[0]: row[1] for row in cursor.fetchall()}

        node_to_idx = self.graph_store.node_to_idx

        # Assign weights to entity nodes based on fact scores
        for rank, f in enumerate(top_k_facts):
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for entity_text in [f[0], f[2]]:  # head and tail
                entity_id = compute_entity_id(normalize_entity_text(entity_text))
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

        passage_node_ids = np.array([node_to_idx[self.passage_node_keys[doc_id]]
                                     for doc_id in sorted_doc_ids], dtype=np.int32)
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
            subgraph_nodes=subgraph_list
        )

        assert len(ppr_sorted_doc_ids) == len(self.passage_node_idxs), \
            f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        # Convert to chunk IDs
        chunk_ids = []
        chunk_scores = []
        for doc_id, score in zip(ppr_sorted_doc_ids, ppr_sorted_doc_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_ids.append(self.passage_node_keys[doc_id])
                chunk_scores.append(score)

        return chunk_ids, chunk_scores, ppr_scores

    def _run_ppr_with_weights(
        self,
        node_weights: np.ndarray,
        damping: float = 0.5,
        subgraph_nodes: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Personalized PageRank with weighted reset probabilities.

        When subgraph_nodes is provided, PPR runs only on the induced subgraph.
        When subgraph_nodes is None, PPR runs on the full graph.

        Args:
            node_weights: Weight for each node (used as reset probabilities)
            damping: Damping factor for PageRank (probability of random jump)
            subgraph_nodes: Optional list of subgraph node indices

        Returns:
            Tuple of (sorted_doc_ids, sorted_doc_scores, pagerank_scores)
        """
        # Clean up weights (remove NaN and negative values)
        node_weights = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)

        if subgraph_nodes is not None and len(subgraph_nodes) > 0:
            # Use induced subgraph
            subgraph_size = len(subgraph_nodes)
            total_size = len(self.graph_store.node_to_idx)
            logger.info(f"Using induced subgraph PPR: {subgraph_size}/{total_size} nodes ({100*subgraph_size/total_size:.1f}%)")

            subgraph = self.graph_store.graph.induced_subgraph(subgraph_nodes)

            # Extract reset probabilities for subgraph nodes
            subgraph_reset = [node_weights[node_idx] for node_idx in subgraph_nodes]

            # Normalize reset probabilities
            reset_sum = sum(subgraph_reset)
            if reset_sum > 0:
                subgraph_reset = [r / reset_sum for r in subgraph_reset]
            else:
                logger.warning("All reset probabilities are zero")
                return np.array([]), np.array([]), np.array([])

            try:
                # Run PPR on subgraph
                subgraph_pagerank = subgraph.personalized_pagerank(
                    damping=damping,
                    directed=False,
                    weights='weight',
                    reset=subgraph_reset,
                    implementation='prpack'
                )

                # Map subgraph scores back to full graph
                pagerank_scores = [0.0] * len(self.graph_store.node_to_idx)
                for i, node_idx in enumerate(subgraph_nodes):
                    pagerank_scores[node_idx] = subgraph_pagerank[i]

            except Exception as e:
                logger.error(f"Subgraph PPR failed: {e}")
                return np.array([]), np.array([]), np.array([])
        else:
            # No subgraph specified, use full graph
            logger.info(f"Using full graph PPR: {len(self.graph_store.node_to_idx)} nodes")

            # Normalize reset probabilities
            reset_sum = node_weights.sum()
            if reset_sum > 0:
                reset_prob = node_weights / reset_sum
            else:
                logger.warning("All reset probabilities are zero")
                return np.array([]), np.array([]), np.array([])

            try:
                # Run PPR on full graph
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

        # Extract and sort passage scores
        pagerank_array = np.array(pagerank_scores)
        doc_scores = pagerank_array[self.passage_node_idxs]

        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids]

        return sorted_doc_ids, sorted_doc_scores, pagerank_array

    def _dense_passage_retrieval_scores(self, query: str) -> np.ndarray:
        """
        Compute dense retrieval scores for all passages using dot product similarity.

        Args:
            query: Query string

        Returns:
            Array of similarity scores for all passages
        """
        query_embedding = self._get_query_embedding(query)

        # Collect passage embeddings
        passage_embeddings_list = []
        for chunk_id in self.passage_node_keys:
            if chunk_id in self.graph_store.chunk_embeddings:
                passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
            else:
                # Use zero embedding for missing chunks
                if passage_embeddings_list:
                    embedding_dim = len(passage_embeddings_list[0])
                else:
                    embedding_dim = 1024
                passage_embeddings_list.append(np.zeros(embedding_dim))

        if not passage_embeddings_list:
            logger.warning("No passage embeddings available")
            return np.zeros(len(self.passage_node_keys))

        passage_embeddings_array = np.array(passage_embeddings_list)

        # Compute dot product similarity
        query_doc_scores = np.dot(passage_embeddings_array, query_embedding)

        return query_doc_scores

    def _dense_passage_retrieval(self, query: str, top_k: int = 10, owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Fallback dense retrieval method (used when graph retrieval fails).

        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            owner_id: Optional owner ID to filter chunks

        Returns:
            List of retrieved Chunk objects
        """
        query_doc_scores = self._dense_passage_retrieval_scores(query)

        top_k_indices = np.argsort(query_doc_scores)[-top_k:][::-1]

        chunk_ids = [self.passage_node_keys[i] for i in top_k_indices if i < len(self.passage_node_keys)]
        scores = [query_doc_scores[i] for i in top_k_indices if i < len(query_doc_scores)]

        chunks = self._convert_to_chunks(chunk_ids, scores, owner_id=owner_id)

        return chunks

    def _convert_to_chunks(self, chunk_ids: List[str], scores: List[float], owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Convert chunk IDs and scores to Chunk objects by querying the database.

        Args:
            chunk_ids: List of chunk IDs
            scores: List of relevance scores
            owner_id: Optional owner ID to filter chunks

        Returns:
            List of Chunk objects with scores in metadata
        """
        chunks = []
        cursor = self.graph_store.conn.cursor()

        for chunk_id, score in zip(chunk_ids, scores):
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
        Rerank facts using LLM to improve relevance.

        Args:
            query: Query string
            query_fact_scores: Scores for all retrieved facts
            fact_ids: List of fact IDs

        Returns:
            Tuple of (reranked_facts, reranked_fact_indices)
        """
        link_top_k = self.config.fact_retrieval_top_k

        # Get top-k candidate facts by score
        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
        candidate_facts = self._get_facts_by_indices(candidate_fact_indices, fact_ids)

        try:
            # Use LLM to rerank and filter facts
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
        Use LLM to select the most relevant facts from candidates.

        Args:
            query: Query string
            candidate_facts: List of candidate fact triples
            candidate_fact_indices: Indices of candidate facts
            len_after_rerank: Number of facts to select

        Returns:
            Tuple of (selected_facts, selected_fact_indices)
        """
        # Format facts for LLM prompt
        facts_text = "\n".join([
            f"{i+1}. {head} - {relation} - {tail}"
            for i, (head, relation, tail) in enumerate(candidate_facts)
        ])

        prompt = f"""Given the query: "{query}"

Select the {len_after_rerank} most relevant facts from the following list:

{facts_text}

Return only the numbers of the selected facts, separated by commas (e.g., "1,3,5").
"""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)

        try:
            # Parse LLM response
            selected_indices = [int(x.strip()) - 1 for x in response.split(",")]
            selected_indices = [i for i in selected_indices if 0 <= i < len(candidate_facts)]

            if not selected_indices:
                raise ValueError("No valid indices in LLM response")

            top_k_facts = [candidate_facts[i] for i in selected_indices[:len_after_rerank]]
            top_k_fact_indices = [candidate_fact_indices[i] for i in selected_indices[:len_after_rerank]]

            return top_k_facts, top_k_fact_indices

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            max_facts = min(len_after_rerank, len(candidate_facts))
            return candidate_facts[:max_facts], candidate_fact_indices[:max_facts]

