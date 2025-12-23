import logging
import numpy as np
import uuid
import json
import threading
from typing import List, Tuple, Set, Dict, TYPE_CHECKING, Optional, Union

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever
from core.utils.owner_guard import is_admin_owner, normalize_owner_id

if TYPE_CHECKING:
    from config.core.retrieval.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jRetrievalConfig

logger = logging.getLogger(__name__)


class PrunedHippoRAGNeo4jRetriever(PrunedHippoRAGRetriever):
    """
    Pruned HippoRAG Retrieval System with Neo4j backend.

    This retriever extends the base PrunedHippoRAGRetriever to work with Neo4j
    as the graph database backend instead of SQLite + igraph.

    Key differences:
    - Graph queries use Neo4j Cypher instead of igraph methods
    - Node mappings are built from Neo4j instead of SQLite
    - Subgraph extraction uses Neo4j queries
    - PageRank still uses igraph (extracted subgraph)
    """

    def __init__(self, config: "PrunedHippoRAGNeo4jRetrievalConfig"):
        """
        Initialize the Pruned HippoRAG Retriever with Neo4j backend.

        Args:
            config: Configuration object containing all retrieval parameters
        """
        self._tls = threading.local()
        # Call parent __init__ but skip some igraph-specific initialization
        self.config = config

        # Build and load graph store (Neo4j version)
        self.graph_store = config.graph_config.build()

        self.embedding_model = self.graph_store.embedding_model

        # Initialize optional LLM client for fact reranking
        self.llm_client = None
        if config.llm_config is not None:
            try:
                self.llm_client = config.llm_config.build()
                logger.info("LLM client initialized for fact filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Will use fallback filtering.")

        # Cache for node mappings to avoid rebuilding on every query
        self._cached_owner_id = None
        self._cached_store_version = None  # Track graph store cache version
        self.passage_node_keys = []
        self.passage_embeddings_array = None

        # PPR backend selection: 'push' (default, faster) or 'igraph' (fallback)
        self.ppr_backend = getattr(config, 'ppr_backend', 'push')

        # Build initial node mappings
        self._build_node_mappings()

        logger.info("Pruned HippoRAG Retrieval System (Neo4j) initialized")
        logger.info(f"  PPR backend: {self.ppr_backend}")
        logger.info(f"  Expansion hops: {config.expansion_hops}")
        logger.info(f"  Include chunk neighbors: {config.include_chunk_neighbors}")
        logger.info(f"  Enable query-aware pruning: {config.enable_pruning}")
        if config.enable_pruning:
            logger.info(f"    Base max neighbors: {config.max_neighbors}")
            logger.info(f"    Query-aware multiplier: {config.query_aware_multiplier}")
        logger.info(f"    Min/Max neighbors: {config.query_aware_min_k}/{config.query_aware_max_k}")

    def _get_tls(self) -> threading.local:
        tls = getattr(self, "_tls", None)
        if tls is None:
            tls = threading.local()
            setattr(self, "_tls", tls)
        return tls

    @property
    def _cached_owner_id(self) -> Optional[str]:
        return getattr(self._get_tls(), "cached_owner_id", None)

    @_cached_owner_id.setter
    def _cached_owner_id(self, value: Optional[str]) -> None:
        setattr(self._get_tls(), "cached_owner_id", value)

    @property
    def _cached_store_version(self) -> Optional[Union[int, str]]:
        return getattr(self._get_tls(), "cached_store_version", None)

    @_cached_store_version.setter
    def _cached_store_version(self, value: Optional[Union[int, str]]) -> None:
        setattr(self._get_tls(), "cached_store_version", value)

    @property
    def passage_embeddings_array(self) -> Optional[np.ndarray]:
        return getattr(self._get_tls(), "passage_embeddings_array", None)

    @passage_embeddings_array.setter
    def passage_embeddings_array(self, value: Optional[np.ndarray]) -> None:
        setattr(self._get_tls(), "passage_embeddings_array", value)
    
    def invalidate_cache(self):
        """Force invalidation of all cached data."""
        self._cached_owner_id = None
        self._cached_store_version = None
        self.passage_node_keys = []
        self.passage_embeddings_array = None

    @staticmethod
    def _owner_to_str(owner_id: Optional[uuid.UUID]) -> Optional[str]:
        return str(owner_id) if owner_id is not None else None

    def _build_node_mappings(self, owner_id: Optional[uuid.UUID] = None, force_rebuild: bool = False):
        """Build mappings between passage nodes and their IDs from Neo4j."""
        current_store_version = self.graph_store.get_cache_version()
        owner_str = self._owner_to_str(owner_id)

        cache_valid = (
            not force_rebuild
            and self._cached_owner_id == owner_str
            and self._cached_store_version == current_store_version
            and self.passage_embeddings_array is not None
        )

        if cache_valid:
            logger.debug("Using cached node mappings for %d passage nodes", len(self.passage_node_keys))
            return

        if self._cached_store_version != current_store_version:
            logger.info(
                "Cache version changed (%s -> %s), rebuilding...",
                self._cached_store_version,
                current_store_version,
            )

        self.passage_node_keys = []

        if owner_str:
            query = """
            MATCH (c:Chunk {owner_id: $owner_id})
            RETURN c.chunk_id AS chunk_id
            ORDER BY c.created_at
            """
            results = self.graph_store._execute_query(query, {'owner_id': owner_str})
        else:
            query = """
            MATCH (c:Chunk)
            RETURN c.chunk_id AS chunk_id
            ORDER BY c.created_at
            """
            results = self.graph_store._execute_query(query)

        self.passage_node_keys = [record['chunk_id'] for record in results]

        passage_embeddings_list = []
        embedding_dim: int | None = None

        read_lock = getattr(self.graph_store, "read_lock", None)
        if callable(read_lock):
            with self.graph_store.read_lock():
                for chunk_id in self.passage_node_keys:
                    if chunk_id in self.graph_store.chunk_embeddings:
                        passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
                    else:
                        if embedding_dim is None:
                            if passage_embeddings_list:
                                embedding_dim = len(passage_embeddings_list[0])
                            else:
                                embedding_dim = self.graph_store.embedding_model.get_embedding_dimension()
                        passage_embeddings_list.append(np.zeros(embedding_dim))
        else:
            for chunk_id in self.passage_node_keys:
                if chunk_id in self.graph_store.chunk_embeddings:
                    passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
                else:
                    if embedding_dim is None:
                        if passage_embeddings_list:
                            embedding_dim = len(passage_embeddings_list[0])
                        else:
                            embedding_dim = self.graph_store.embedding_model.get_embedding_dimension()
                    passage_embeddings_list.append(np.zeros(embedding_dim))

        if passage_embeddings_list:
            self.passage_embeddings_array = np.array(passage_embeddings_list, dtype=np.float32)
        else:
            self.passage_embeddings_array = np.array([], dtype=np.float32)

        self._cached_owner_id = owner_str
        self._cached_store_version = current_store_version

        logger.info("Built mappings for %d passage nodes", len(self.passage_node_keys))

    def _get_pruned_neighbors_by_weight(
        self,
        node_id: str,
        entity_relevance_scores: dict = None,
        owner_id: Optional[uuid.UUID] = None
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
        neighbors_with_weights = self.graph_store.get_neighbors_with_weights(
            node_id,
            owner_id=self._owner_to_str(owner_id)
        )

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
        owner_id: Optional[uuid.UUID] = None
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
            neighbors = self.graph_store.get_neighbors_with_weights(
                entity_id,
                owner_id=owner_str
            )
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
            batch_neighbors = self.graph_store.get_batch_neighbors_with_weights(
                current_layer_list,
                owner_id=owner_str
            )

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
                entity_batch_neighbors = self.graph_store.get_batch_neighbors_with_weights(
                    next_layer_list,
                    owner_id=owner_str
                )

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

            logger.info(f"Hop {hop}: {len(current_layer)} nodes, pruned {total_neighbors_before_pruning} → {total_neighbors_after_pruning} neighbors")

            current_layer = next_layer
            if not current_layer:
                break

        return subgraph_nodes, subgraph_chunk_ids

    def _run_ppr_with_weights(
        self,
        node_weights: dict,
        damping: float = 0.5,
        subgraph_nodes: Set[str] = None,
        owner_id: Optional[uuid.UUID] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run Personalized PageRank with weighted reset probabilities.

        This method supports two backends:
        - 'push': Fast push-based PPR on cached graph (default, recommended)
        - 'igraph': Traditional igraph-based PPR (fallback)

        Args:
            node_weights: Dict mapping node IDs to weights (used as reset probabilities)
            damping: Damping factor for PageRank (probability of random jump)
            subgraph_nodes: Optional set of subgraph node IDs

        Returns:
            Tuple of (sorted_doc_ids, sorted_doc_scores, pagerank_scores_dict)
        """
        if subgraph_nodes is None or len(subgraph_nodes) == 0:
            logger.warning("No subgraph nodes provided")
            return np.array([]), np.array([]), {}

        # Normalize reset probabilities
        reset_sum = sum(node_weights.values())
        if reset_sum == 0:
            logger.warning("All reset probabilities are zero")
            return np.array([]), np.array([]), {}

        normalized_reset = {nid: w / reset_sum for nid, w in node_weights.items()}

        # Choose PPR backend
        if self.ppr_backend == 'push':
            pagerank_scores_dict = self._run_ppr_push(subgraph_nodes, normalized_reset, damping, owner_id=owner_id)
        else:
            pagerank_scores_dict = self._run_ppr_igraph(subgraph_nodes, normalized_reset, damping, owner_id=owner_id)

        if not pagerank_scores_dict:
            logger.warning("PPR returned no scores")
            return np.array([]), np.array([]), {}

        # Extract and sort passage scores
        doc_scores = []
        doc_ids = []
        for i, chunk_id in enumerate(self.passage_node_keys):
            if chunk_id in pagerank_scores_dict:
                doc_scores.append(pagerank_scores_dict[chunk_id])
                doc_ids.append(i)

        doc_scores = np.array(doc_scores)
        doc_ids = np.array(doc_ids)

        sorted_indices = np.argsort(doc_scores)[::-1]
        sorted_doc_ids = doc_ids[sorted_indices]
        sorted_doc_scores = doc_scores[sorted_indices]

        return sorted_doc_ids, sorted_doc_scores, pagerank_scores_dict

    def _run_ppr_push(
        self,
        subgraph_nodes: Set[str],
        reset: Dict[str, float],
        damping: float,
        owner_id: Optional[uuid.UUID] = None
    ) -> Dict[str, float]:
        """
        Run PPR using push-based algorithm on cached graph.

        Args:
            subgraph_nodes: Set of node IDs in subgraph
            reset: Normalized reset distribution
            damping: Damping factor (alpha)

        Returns:
            Dictionary mapping node_id -> PageRank score
        """
        try:
            ppr_scores = self.graph_store.compute_ppr_push(
                subgraph_nodes=subgraph_nodes,
                reset=reset,
                alpha=damping,
                epsilon=1e-6,
                owner_id=self._owner_to_str(owner_id)
            )
            logger.info(f"Push-based PPR completed: {len(ppr_scores)} nodes with non-zero scores")
            return ppr_scores
        except Exception as e:
            logger.error(f"Push-based PPR failed: {e}, falling back to igraph")
            return self._run_ppr_igraph(subgraph_nodes, reset, damping, owner_id=owner_id)

    def _run_ppr_igraph(
        self,
        subgraph_nodes: Set[str],
        reset: Dict[str, float],
        damping: float,
        owner_id: Optional[uuid.UUID] = None
    ) -> Dict[str, float]:
        """
        Run PPR using igraph (fallback method).

        Args:
            subgraph_nodes: Set of node IDs in subgraph
            reset: Normalized reset distribution
            damping: Damping factor

        Returns:
            Dictionary mapping node_id -> PageRank score
        """
        # Extract subgraph from cache (faster) or Neo4j (fallback)
        graph, _, idx_to_node = self.graph_store.extract_subgraph_from_cache(
            subgraph_nodes,
            owner_id=self._owner_to_str(owner_id)
        )

        if graph.vcount() == 0:
            logger.warning("Empty subgraph extracted")
            return {}

        logger.info(f"Using igraph PPR: {graph.vcount()} nodes, {graph.ecount()} edges")

        # Build reset probabilities for subgraph nodes
        subgraph_reset = [reset.get(idx_to_node[i], 0.0) for i in range(graph.vcount())]

        try:
            # Run PPR on subgraph
            subgraph_pagerank = graph.personalized_pagerank(
                damping=damping,
                directed=False,
                weights='weight',
                reset=subgraph_reset,
                implementation='prpack'
            )

            # Map subgraph scores back to node IDs
            pagerank_scores_dict = {}
            for i, score in enumerate(subgraph_pagerank):
                node_id = idx_to_node[i]
                pagerank_scores_dict[node_id] = score

            return pagerank_scores_dict

        except Exception as e:
            logger.error(f"igraph PPR failed: {e}")
            return {}

    def _graph_search_on_subgraph(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[int],
        subgraph_nodes: Set[str],
        owner_id: Optional[uuid.UUID] = None
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
        entity_to_chunk_count = self.graph_store.get_batch_entity_chunk_counts_from_cache(
            list(entity_ids_in_facts),
            owner_id=owner_str
        )

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
            owner_id=owner_id
        )

        # Convert to chunk IDs
        chunk_ids = []
        chunk_scores = []
        for doc_id, score in zip(ppr_sorted_doc_ids, ppr_sorted_doc_scores):
            if doc_id < len(self.passage_node_keys):
                chunk_ids.append(self.passage_node_keys[doc_id])
                chunk_scores.append(score)

        return chunk_ids, chunk_scores, ppr_scores_dict

    def retrieve(self, query: str, top_k: int = 10, return_subgraph_info: bool = False, owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Main retrieval method implementing the Pruned HippoRAG algorithm with Neo4j backend.

        Retrieval pipeline:
        1. Retrieve relevant facts using FAISS dense retrieval
        2. Optionally rerank facts using LLM
        3. Extract seed entities from top facts
        4. Expand subgraph around seed entities (using Neo4j)
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

        if owner_id is None:
            logger.warning("Owner ID is required for graph retrieval; returning empty results")
            return []

        normalized_owner = normalize_owner_id(owner_id)
        if normalized_owner is None:
            logger.warning("Unable to normalize owner_id '%s'; returning empty results", owner_id)
            return []

        if is_admin_owner(owner_id):
            logger.info("Admin owner detected, retrieving across all owners")
            owner_filter = None
        else:
            owner_filter = owner_id

        # Rebuild node mappings for the current owner
        self._build_node_mappings(owner_id=owner_filter)

        # Step 1: Retrieve relevant facts
        query_fact_scores, fact_ids = self._get_fact_scores_faiss(query, owner_id=owner_filter)

        if query_fact_scores is None or len(query_fact_scores) == 0:
            logger.warning("No facts found, falling back to dense retrieval")
            return self._dense_passage_retrieval(query, top_k, owner_id=owner_filter)

        # Step 2: Rerank facts (optional)
        if self.config.enable_llm_reranking and self.llm_client:
            top_k_facts, top_k_fact_indices = self._rerank_facts(
                query,
                query_fact_scores,
                fact_ids,
                owner_id=owner_filter
            )
        else:
            link_top_k = self.config.fact_retrieval_top_k
            top_k_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
            top_k_facts = self._get_facts_by_indices(top_k_fact_indices, fact_ids, owner_id=owner_filter)

        if not top_k_facts:
            logger.warning("No facts after reranking, falling back to dense retrieval")
            return self._dense_passage_retrieval(query, top_k, owner_id=owner_filter)

        logger.info(f"Selected {len(top_k_facts)} facts after LLM filtering")

        # Step 3: Extract seed entities from facts
        seed_entity_ids = self._extract_entity_ids_from_facts(top_k_facts)

        if not seed_entity_ids:
            logger.warning("No seed entities found, falling back to dense retrieval")
            return self._dense_passage_retrieval(query, top_k, owner_id=owner_filter)

        logger.info(f"Extracted {len(seed_entity_ids)} seed entities from {len(top_k_facts)} facts")

        # Step 4: Compute entity relevance scores for query-aware pruning
        entity_relevance_scores = None
        if self.config.enable_pruning:
            entity_relevance_scores = self._compute_entity_relevance_scores(
                seed_entity_ids,
                top_k_facts,
                query_fact_scores,
                top_k_fact_indices,
                owner_id=owner_filter
            )
            logger.info(f"[Query-Aware] Computed relevance scores for {len(entity_relevance_scores)} entities")

        # Step 5: Expand subgraph around seed entities (using Neo4j)
        subgraph_nodes, subgraph_chunk_ids = self._expand_subgraph(
            seed_entity_ids,
            entity_relevance_scores=entity_relevance_scores,
                owner_id=owner_filter
            )

        logger.info(f"Subgraph: {len(subgraph_nodes)} nodes, {len(subgraph_chunk_ids)} chunks")

        # Step 6: Perform graph search using Personalized PageRank
        sorted_doc_ids, sorted_doc_scores, ppr_scores_dict = self._graph_search_on_subgraph(
            query,
            query_fact_scores,
            top_k_facts,
            top_k_fact_indices,
            subgraph_nodes,
            owner_id=owner_filter
        )

        # Step 7: Convert to Chunk objects
        chunks = self._convert_to_chunks(sorted_doc_ids[:top_k], sorted_doc_scores[:top_k], owner_id=owner_filter)

        # Optionally attach subgraph information for visualization
        if return_subgraph_info and chunks:
            node_to_ppr_score = ppr_scores_dict  # Already a dict in Neo4j version

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

    def _convert_to_chunks(self, chunk_ids: List[str], scores: List[float], owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Convert chunk IDs and scores to Chunk objects by querying Neo4j (batch query).

        Args:
            chunk_ids: List of chunk IDs
            scores: List of relevance scores
            owner_id: Optional owner ID to filter chunks

        Returns:
            List of Chunk objects with scores in metadata
        """
        if not chunk_ids:
            return []

        # Batch query all chunks at once
        if owner_id is not None:
            query = """
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunk_ids AND c.owner_id = $owner_id
            RETURN c.chunk_id AS chunk_id, c.content AS content, c.owner_id AS owner_id, c.metadata AS metadata
            """
            results = self.graph_store._execute_query(query, {
                'chunk_ids': chunk_ids,
                'owner_id': str(owner_id)
            })
        else:
            query = """
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunk_ids
            RETURN c.chunk_id AS chunk_id, c.content AS content, c.owner_id AS owner_id, c.metadata AS metadata
            """
            results = self.graph_store._execute_query(query, {'chunk_ids': chunk_ids})

        # Build chunk_id -> chunk data mapping
        chunk_data_map = {}
        for record in results:
            chunk_id = record['chunk_id']
            owner_value = self.graph_store._restore_owner_id(record.get('owner_id'))
            chunk_data_map[chunk_id] = {
                'content': record['content'],
                'owner_id': owner_value,
                'metadata': record['metadata']
            }

        # Create Chunk objects in the same order as chunk_ids
        chunks = []
        for chunk_id, score in zip(chunk_ids, scores):
            if chunk_id in chunk_data_map:
                data = chunk_data_map[chunk_id]

                # Parse metadata
                try:
                    metadata = json.loads(data['metadata']) if data['metadata'] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Add score to metadata
                metadata['score'] = float(score)

                # Create Chunk object with restored owner type
                owner_value = data.get('owner_id')
                owner_field: Optional[Union[str, uuid.UUID]] = None
                if owner_value:
                    try:
                        owner_field = uuid.UUID(owner_value)
                    except (ValueError, TypeError):
                        owner_field = owner_value

                chunk = Chunk(
                    id=chunk_id,
                    content=data['content'],
                    owner_id=owner_field,
                    metadata=metadata
                )
                chunks.append(chunk)

        return chunks

    def _get_fact_scores_faiss(self, query: str, owner_id: Optional[uuid.UUID] = None) -> Tuple[np.ndarray, List[str]]:
        scores, fact_ids = super()._get_fact_scores_faiss(query, owner_id=owner_id)

        if owner_id is None or scores is None or len(scores) == 0:
            return scores, fact_ids

        owner_str = self._owner_to_str(owner_id)
        docstore = self.graph_store.fact_faiss_db.docstore

        filtered_scores = []
        filtered_ids = []

        for idx, fact_id in enumerate(fact_ids):
            chunk = docstore.get(fact_id)
            if not chunk:
                continue

            fact_owner = getattr(chunk, 'owner_id', None)
            if fact_owner is None and chunk.metadata:
                fact_owner = chunk.metadata.get('owner_id')

            if fact_owner is None:
                continue

            if str(fact_owner) == owner_str:
                score_value = scores[idx] if isinstance(scores, np.ndarray) else scores[idx]
                filtered_scores.append(float(score_value))
                filtered_ids.append(fact_id)

        if not filtered_scores:
            return np.array([]), []

        dtype = scores.dtype if isinstance(scores, np.ndarray) else np.float32
        return np.array(filtered_scores, dtype=dtype), filtered_ids

    def _get_facts_by_indices(
        self,
        indices: List[int],
        fact_ids: List[str],
        owner_id: Optional[uuid.UUID] = None
    ) -> List[Tuple[str, str, str, Optional[str]]]:
        """
        Retrieve fact triples from FAISS docstore (no database query needed).

        Args:
            indices: List of indices into fact_ids
            fact_ids: List of fact IDs

        Returns:
            List of fact triples (head, relation, tail)
        """
        import ast
        facts: List[Tuple[str, str, str, Optional[str]]] = []
        owner_str = self._owner_to_str(owner_id)

        for idx in indices:
            if idx < len(fact_ids):
                fact_id = fact_ids[idx]
                # Retrieve fact from FAISS docstore (contains full Chunk with fact content)
                if fact_id in self.graph_store.fact_faiss_db.docstore:
                    chunk = self.graph_store.fact_faiss_db.docstore[fact_id]

                    fact_owner = getattr(chunk, 'owner_id', None)
                    if fact_owner is None and chunk.metadata:
                        fact_owner = chunk.metadata.get('owner_id')

                    if owner_str and str(fact_owner) != owner_str:
                        continue

                    fact_content = chunk.content
                    parsed_fact: Optional[Tuple[str, str, str]] = None

                    if isinstance(fact_content, tuple) and len(fact_content) == 3:
                        parsed_fact = (fact_content[0], fact_content[1], fact_content[2])
                    elif isinstance(fact_content, str):
                        # Try to parse as Python literal (tuple string representation)
                        try:
                            parsed = ast.literal_eval(fact_content)
                            if isinstance(parsed, tuple) and len(parsed) == 3:
                                parsed_fact = (parsed[0], parsed[1], parsed[2])
                            else:
                                parts = fact_content.split(' | ')
                                if len(parts) == 3:
                                    parsed_fact = (parts[0], parts[1], parts[2])
                        except Exception:
                            parts = fact_content.split(' | ')
                            if len(parts) == 3:
                                parsed_fact = (parts[0], parts[1], parts[2])
                    elif isinstance(fact_content, (list, np.ndarray)) and len(fact_content) >= 3:
                        parsed_fact = (fact_content[0], fact_content[1], fact_content[2])

                    if parsed_fact:
                        facts.append((parsed_fact[0], parsed_fact[1], parsed_fact[2], fact_owner))

        return facts

    def _extract_entity_ids_from_facts(self, facts: List[Tuple[str, str, str, Optional[str]]]) -> Set[str]:
        """
        Extract unique entity IDs from fact triples using Neo4j.

        Args:
            facts: List of fact triples (head, relation, tail)

        Returns:
            Set of entity IDs appearing in the facts
        """
        from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing

        entity_ids = set()

        for head_name, _, tail_name, fact_owner in facts:
            owner_str = self._owner_to_str(fact_owner)
            head_normalized = text_processing(head_name)
            tail_normalized = text_processing(tail_name)
            head_id = compute_mdhash_id(head_normalized, prefix='entity-', owner_id=owner_str)
            tail_id = compute_mdhash_id(tail_normalized, prefix='entity-', owner_id=owner_str)
            entity_ids.add(head_id)
            entity_ids.add(tail_id)

        return entity_ids

    def _dense_passage_retrieval_scores(self, query: str) -> np.ndarray:
        """
        Compute dense retrieval scores for all passages using dot product similarity.

        Optimized version using pre-computed passage embeddings array.
        Supports float16 embeddings and normalized embeddings for cosine similarity.

        Args:
            query: Query string

        Returns:
            Array of similarity scores for all passages
        """
        query_embedding = self._get_query_embedding(query)

        # Normalize query embedding if chunk embeddings are normalized
        if self.graph_store.normalize_chunk_embeddings:
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        if self.passage_embeddings_array.size == 0:
            logger.warning("No passage embeddings available")
            return np.zeros(len(self.passage_node_keys))

        # Fast dot product using numpy
        # If embeddings are float16, convert query to float16 for consistency
        if self.passage_embeddings_array.dtype == np.float16:
            query_embedding = query_embedding.astype(np.float16)

        doc_scores = np.dot(self.passage_embeddings_array, query_embedding)

        # Convert back to float32 for downstream processing
        if doc_scores.dtype == np.float16:
            doc_scores = doc_scores.astype(np.float32)

        return doc_scores

    def _compute_entity_relevance_scores(
        self,
        seed_entity_ids: Set[str],
        top_k_facts: List[Tuple],
        query_fact_scores: np.ndarray,
        top_k_fact_indices: List[int],
        owner_id: Optional[uuid.UUID] = None
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
            Dictionary mapping entity IDs to relevance scores (not graph indices)
        """
        from collections import defaultdict
        from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing

        # Collect fact scores for each entity
        entity_to_fact_scores = defaultdict(list)

        for fact_idx, fact in zip(top_k_fact_indices, top_k_facts):
            fact_owner = self._owner_to_str(fact[3])
            head_name = text_processing(fact[0])
            tail_name = text_processing(fact[2])
            head_id = compute_mdhash_id(head_name, prefix='entity-', owner_id=fact_owner)
            tail_id = compute_mdhash_id(tail_name, prefix='entity-', owner_id=fact_owner)

            fact_score = float(query_fact_scores[fact_idx]) if query_fact_scores.ndim > 0 else float(query_fact_scores)

            entity_to_fact_scores[head_id].append(fact_score)
            entity_to_fact_scores[tail_id].append(fact_score)

        # Compute average relevance score for each entity
        entity_relevance_scores = {}

        for entity_id in seed_entity_ids:
            if entity_id in entity_to_fact_scores:
                avg_score = float(np.mean(entity_to_fact_scores[entity_id]))
                # In Neo4j version, we use entity_id directly instead of graph index
                entity_relevance_scores[entity_id] = avg_score

        if entity_relevance_scores:
            logger.info(f"[Query-Aware] Entity relevance scores: min={min(entity_relevance_scores.values()):.3f}, "
                       f"max={max(entity_relevance_scores.values()):.3f}, "
                       f"avg={np.mean(list(entity_relevance_scores.values())):.3f}")

        return entity_relevance_scores
