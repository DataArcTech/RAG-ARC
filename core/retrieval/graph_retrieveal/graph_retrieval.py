"""
Graph-based Retrieval System for RAG-ARC

This module implements a sophisticated graph-based retrieval system that combines:
1. Parallel candidate recall (semantic search)
2. Subgraph construction and pruning
3. Personalized PageRank (PPR)
4. Chunk backtracking and graph scoring
5. Fusion and ranking

Based on the design document specifications.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
import json
import numpy as np
import math

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_retrieveal.base import BaseGraphRetriever
from core.retrieval.graph_retrieveal.models import CandidateResult, ChunkScore, PPRResult, SubgraphResult
from core.retrieval.graph_retrieveal.metadata_keys import (
    GRAPH_RETRIEVAL_CHUNK_SOURCE_KEY,
    GRAPH_RETRIEVAL_FALLBACK_KEY,
)
from core.retrieval.graph_retrieveal.ppr import compute_personalized_pagerank as _compute_personalized_pagerank
from core.retrieval.graph_retrieveal.query_executor import GraphQueryExecutor, build_graph_query_executor
from core.retrieval.graph_retrieveal.similarity_edges import filter_similarity_neighbors
from core.retrieval.graph_retrieveal.llm_selection import parse_ranked_choice_indices
from core.prompts.graph_retrieval import SEED_ENTITY_FILTER_USER_PROMPT
from core.utils.llm_json import call_llm_json_with_retry_sync

if TYPE_CHECKING:
    from config.core.retrieval.graph_retrieval_config import GraphRetrievalConfig

logger = logging.getLogger(__name__)


class GraphRetrieval(BaseGraphRetriever):
    """
    Graph-based Retrieval System

    Implements the 5-step retrieval process:
    1. Parallel Candidate Recall
    2. Subgraph Construction and Pruning
    3. Personalized PageRank
    4. Chunk Backtracking and Graph Scoring
    5. Fusion and Ranking
    """

    def __init__(self, config: "GraphRetrievalConfig"):
        """Initialize Graph Retrieval System"""
        super().__init__(config)

        # Initialize the graph store (supports both Neo4j and NetworkX)
        self.graph_store = config.graph_config.build()
        self.embedding_model = config.embedding_config.build()

        # Determine graph store type for query adaptation
        self.graph_store_type = config.graph_config.type
        logger.info(f"Initialized graph store type: {self.graph_store_type}")
        self._query_executor: GraphQueryExecutor = build_graph_query_executor(
            graph_store=self.graph_store,
            graph_store_type=self.graph_store_type,
        )

        # Initialize LLM for entity filtering (optional)
        self.llm = None
        if config.llm_config is not None:
            try:
                self.llm = config.llm_config.build()
                logger.info("LLM initialized for entity filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Will use rule-based filtering.")

        logger.info("Graph Retrieval System initialized")

    def _execute_graph_query(self, query_type: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute graph query with adaptation for different graph stores"""
        return self._query_executor.execute(query_type, params)

    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        """
        Main retrieval method - returns List[Chunk]

        Args:
            query: Natural language query
            top_k: Number of top documents to return

        Returns:
            List of Chunk objects sorted by relevance
        """
        logger.info(f"Starting retrieval for query: {query}")

        # Step 1: Parallel Candidate Recall
        candidates = self.parallel_candidate_recall(query)
        logger.info(f"Retrieved {len(candidates.entity_candidates)} entity candidates, "
                   f"{len(candidates.chunk_candidates)} chunk candidates")

        # Step 2: Subgraph Construction and Pruning
        subgraph = self.construct_subgraph(candidates.entity_candidates, query)
        logger.info(f"Constructed subgraph with {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")

        # Step 3: Personalized PageRank
        ppr_result = self.compute_personalized_pagerank(subgraph, candidates.entity_candidates)
        logger.info(f"Computed PPR scores for {len(ppr_result.scores)} entities")

        # Step 4: Chunk Backtracking and Graph Scoring
        chunk_scores = self.compute_chunk_scores(query, ppr_result, candidates.chunk_candidates)
        logger.info(f"Computed scores for {len(chunk_scores)} chunks")

        # Step 5: Fusion and Ranking
        final_scores = self.fusion_and_ranking(chunk_scores)
        logger.info(f"Final ranking completed, returning top {min(top_k, len(final_scores))} chunks")

        # Convert ChunkScore objects to Chunk objects
        chunks = []
        for chunk_score in final_scores[:top_k]:
            # Get the original chunk
            try:
                chunk_list = self.graph_store.get_by_ids([chunk_score.chunk_id])
                if chunk_list:
                    chunk = chunk_list[0]

                    # Add scores to metadata
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata.update({
                        'score': chunk_score.final_score,
                        'graph_score': chunk_score.graph_score,
                        'embedding_score': chunk_score.embedding_score,
                        'mentioned_entities': chunk_score.mentioned_entities,
                        GRAPH_RETRIEVAL_CHUNK_SOURCE_KEY: "graph_store",
                    })

                    chunks.append(chunk)
                else:
                    # Create a chunk from chunk score if original not found
                    chunk = Chunk(
                        id=chunk_score.chunk_id,
                        content=chunk_score.content,
                        metadata={
                            'score': chunk_score.final_score,
                            'graph_score': chunk_score.graph_score,
                            'embedding_score': chunk_score.embedding_score,
                            'mentioned_entities': chunk_score.mentioned_entities,
                            GRAPH_RETRIEVAL_CHUNK_SOURCE_KEY: "chunk_score",
                            GRAPH_RETRIEVAL_FALLBACK_KEY: {
                                "reason": "graph_store_miss",
                                "chunk_id": chunk_score.chunk_id,
                            },
                        }
                    )
                    chunks.append(chunk)

            except Exception as e:
                logger.warning(
                    "Could not retrieve chunk %s from graph_store.get_by_ids: %s",
                    chunk_score.chunk_id,
                    e,
                    exc_info=True,
                )
                # Create a fallback chunk
                chunk = Chunk(
                    id=chunk_score.chunk_id,
                    content=chunk_score.content,
                    metadata={
                        'score': chunk_score.final_score,
                        'graph_score': chunk_score.graph_score,
                        'embedding_score': chunk_score.embedding_score,
                        'mentioned_entities': chunk_score.mentioned_entities,
                        GRAPH_RETRIEVAL_CHUNK_SOURCE_KEY: "fallback_error",
                        GRAPH_RETRIEVAL_FALLBACK_KEY: {
                            "reason": "graph_store_exception",
                            "chunk_id": chunk_score.chunk_id,
                            "exception_type": e.__class__.__name__,
                            "message": str(e),
                        },
                    }
                )
                chunks.append(chunk)

        return chunks

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[ChunkScore]:
        """
        Internal retrieval method that returns ChunkScore objects for debugging

        Args:
            query: Natural language query
            top_k: Number of top chunks to return

        Returns:
            List of ChunkScore objects sorted by final score
        """
        logger.info(f"Starting retrieval for query: {query}")

        # Step 1: Parallel Candidate Recall
        candidates = self.parallel_candidate_recall(query)
        logger.info(f"Retrieved {len(candidates.entity_candidates)} entity candidates, "
                   f"{len(candidates.chunk_candidates)} chunk candidates")

        # Step 2: Subgraph Construction and Pruning
        subgraph = self.construct_subgraph(candidates.entity_candidates, query)
        logger.info(f"Constructed subgraph with {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")

        # Step 3: Personalized PageRank
        ppr_result = self.compute_personalized_pagerank(subgraph, candidates.entity_candidates)
        logger.info(f"Computed PPR scores for {len(ppr_result.scores)} entities")

        # Step 4: Chunk Backtracking and Graph Scoring
        chunk_scores = self.compute_chunk_scores(query, ppr_result, candidates.chunk_candidates)
        logger.info(f"Computed scores for {len(chunk_scores)} chunks")

        # Step 5: Fusion and Ranking
        final_scores = self.fusion_and_ranking(chunk_scores)
        logger.info(f"Final ranking completed, returning top {min(top_k, len(final_scores))} chunks")

        return final_scores[:top_k]
    

    def parallel_candidate_recall(self, query: str) -> CandidateResult:
        """
        Step 1: Parallel Candidate Recall

        Perform semantic search for both entities and chunks
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Query → Entity: Get top k2 entities
        entity_candidates = self.semantic_search_entities(query_embedding, self.config.k2_entities)
        
        # Query → Chunk: Get top k1 chunks  
        chunk_candidates = self.semantic_search_chunks(query_embedding, self.config.k1_chunks)
        
        return CandidateResult(
            entity_candidates=entity_candidates,
            chunk_candidates=chunk_candidates
        )
    
    def semantic_search_entities(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Semantic search for entities using vector similarity"""
        # Enhanced entity search with better filtering and scoring

        # Get entities with embeddings, including additional metadata for better scoring
        results = self._execute_graph_query("semantic_search_entities")
        candidates = []

        for result in results:
            entity_embedding = result['embedding']
            if entity_embedding and len(entity_embedding) > 0:
                # Compute cosine similarity
                similarity = self.cosine_similarity(query_embedding, entity_embedding)

                # Apply additional scoring factors
                mention_count = result.get('mention_count', 0)
                entity_type = result.get('entity_type', 'Entity')

                # Boost score based on mention frequency (popularity)
                popularity_boost = self._popularity_boost(int(mention_count or 0))

                # Final similarity score
                final_similarity = similarity + popularity_boost

                candidates.append({
                    'entity_id': result['entity_id'],
                    'entity_name': result['entity_name'],
                    'entity_type': entity_type,
                    'similarity': final_similarity,
                    'raw_similarity': similarity,
                    'mention_count': mention_count,
                    'attributes': json.loads(result.get('attributes') or '{}')
                })

        # Sort by final similarity and return top k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:top_k]

    def _popularity_boost(self, mention_count: int) -> float:
        if mention_count <= 0:
            return 0.0
        divisor = float(getattr(self.config, "mention_count_boost_log_divisor", 10.0) or 10.0)
        if divisor <= 0:
            divisor = 10.0
        cap = float(getattr(self.config, "mention_count_boost_max", 0.1) or 0.1)
        cap = max(0.0, cap)
        return min(math.log(mention_count + 1) / divisor, cap)


    def semantic_search_chunks(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Semantic search for chunk chunks using vector similarity"""
        # Enhanced chunk search with metadata and entity information

        results = self._execute_graph_query("semantic_search_chunks")
        candidates = []

        for result in results:
            chunk_embedding = result['embedding']
            if chunk_embedding and len(chunk_embedding) > 0:
                # Compute cosine similarity
                similarity = self.cosine_similarity(query_embedding, chunk_embedding)

                # Apply additional scoring factors
                entity_count = result.get('entity_count', 0)

                # Boost score based on entity richness
                entity_boost = min(math.log(entity_count + 1) / 20, 0.05)


                # Final similarity score
                final_similarity = similarity + entity_boost

                candidates.append({
                    'chunk_id': result['chunk_id'],
                    'content': result['content'],
                    'similarity': final_similarity,
                    'raw_similarity': similarity,
                    'entity_count': entity_count,
                    'entity_names': result.get('entity_names', []),
                    'metadata': json.loads(result.get('metadata') or '{}')
                })

        # Sort by final similarity and return top k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:top_k]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def construct_subgraph(self, entity_candidates: List[Dict[str, Any]], query: str) -> SubgraphResult:
        """
        Step 2: Subgraph Construction and Pruning

        Build Entity-Only subgraph with hop limits and beam search
        """
        # Filter seed entities using LLM or rule-based approach
        seed_entities = self.filter_seed_entities(entity_candidates, query)

        # Initialize subgraph
        nodes = {}
        edges = []
        visited = set()

        # BFS with beam search
        current_level = seed_entities.copy()

        for hop in range(self.config.max_hops):
            if not current_level:
                break

            next_level = []

            # Process current level entities
            for entity_id in current_level:
                if entity_id in visited:
                    continue

                visited.add(entity_id)

                # Get entity data
                entity_data = self.get_entity_data(entity_id)
                if entity_data:
                    nodes[entity_id] = entity_data

                # Get neighbors
                neighbors = self.get_entity_neighbors(entity_id)
                neighbors = filter_similarity_neighbors(
                    neighbors,
                    hop=hop,
                    relation_name=getattr(self.config, "similarity_edge_relation", "SIMILAR_TO"),
                    max_hops=int(getattr(self.config, "similarity_edge_max_hops", 1)),
                    min_similarity=float(getattr(self.config, "similarity_edge_min_similarity", 0.0)),
                    max_per_node=int(getattr(self.config, "similarity_edge_max_per_node", 0)),
                )

                # Add edges and collect next level candidates
                for neighbor in neighbors:
                    if not isinstance(neighbor, dict):
                        continue
                    neighbor_id = neighbor.get("neighbor_id")
                    relation_type = neighbor.get("relation_type")
                    if not neighbor_id or not relation_type:
                        continue
                    if neighbor_id not in visited:
                        edges.append((entity_id, relation_type, neighbor_id))
                        next_level.append(neighbor_id)

            # Apply beam search: keep only top candidates for next level
            if len(next_level) > self.config.beam_size:
                # Score candidates by specificity and path length
                scored_candidates = []
                for candidate in next_level:
                    score = self.compute_candidate_priority(candidate, hop + 1)
                    scored_candidates.append((candidate, score))

                # Sort and keep top beam_size candidates
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                next_level = [candidate for candidate, _ in scored_candidates[:self.config.beam_size]]

            current_level = next_level

        return SubgraphResult(
            nodes=nodes,
            edges=edges,
            seed_entities=seed_entities
        )

    def filter_seed_entities(self, entity_candidates: List[Dict[str, Any]], query: str) -> List[str]:
        """Filter entity candidates to get high-relevance seed entities using LLM or rule-based approach"""

        if not entity_candidates:
            return []

        # Try LLM-based filtering first if available
        if self.llm is not None:
            try:
                # Prepare entity information for LLM (name, type, attributes only)
                entity_info = []
                for i, candidate in enumerate(entity_candidates):
                    attributes = candidate.get('attributes', {})
                    # Format attributes as readable text
                    attr_text = ", ".join([f"{k}: {v}" for k, v in attributes.items()]) if attributes else "None"

                    entity_info.append({
                        'index': i + 1,
                        'entity_id': candidate['entity_id'],
                        'entity_name': candidate.get('entity_name', 'Unknown'),
                        'entity_type': candidate.get('entity_type', 'Entity'),
                        'attributes': attr_text
                    })

                entities_text = "\n".join(
                    [
                        f"{entity['index']}. {entity['entity_name']} (type={entity['entity_type']}; attributes={entity['attributes']})"
                        for entity in entity_info
                    ]
                )
                prompt = SEED_ENTITY_FILTER_USER_PROMPT.format(query=str(query), entities_text=entities_text)

                # Get LLM response with centralized JSON retry.
                messages = [{"role": "user", "content": prompt}]
                payload = call_llm_json_with_retry_sync(
                    llm_connector=self.llm,
                    messages=messages,
                    expected=None,
                    return_raw=True,
                )
                if isinstance(payload, tuple):
                    parsed_payload, raw_response = payload
                    response_obj = parsed_payload if parsed_payload is not None else raw_response
                else:
                    response_obj = payload

                # Parse LLM response to get selected entity indices
                try:
                    parsed = parse_ranked_choice_indices(
                        response_obj,
                        candidate_count=len(entity_candidates),
                        k=5,
                        one_based=True,
                    )
                    if not parsed:
                        logger.warning("Failed to parse LLM response, using top 2 entities as fallback")
                        parsed = [0, 1] if len(entity_candidates) >= 2 else ([0] if len(entity_candidates) >= 1 else [])

                except Exception as e:
                    logger.warning(f"Error parsing LLM response: {e}. Using top 2 entities as fallback.")
                    parsed = [0, 1] if len(entity_candidates) >= 2 else ([0] if len(entity_candidates) >= 1 else [])

                # Convert indices to entity IDs
                seed_entities = [entity_candidates[i]['entity_id'] for i in parsed]

                logger.info(f"LLM filtered {len(entity_candidates)} candidates to {len(seed_entities)} seed entities")
                return seed_entities

            except Exception as e:
                logger.warning(f"LLM filtering failed: {e}. Falling back to rule-based filtering.")

        # Fallback to rule-based filtering
        # Dynamic threshold based on score distribution
        similarities = [c['similarity'] for c in entity_candidates]
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # Use mean + 0.5*std as threshold, but at least 0.3
        dynamic_threshold = max(0.3, mean_sim + 0.5 * std_sim)

        seed_entities = []
        for candidate in entity_candidates:
            # Primary filter: similarity threshold
            if candidate['similarity'] >= dynamic_threshold:
                seed_entities.append(candidate['entity_id'])
            # Secondary filter: high-mention entities with decent similarity
            elif (candidate['similarity'] >= 0.2 and
                  candidate.get('mention_count', 0) >= 3):
                seed_entities.append(candidate['entity_id'])

        # Ensure we have at least some seed entities
        if not seed_entities and entity_candidates:
            # Take top 2 entities as fallback
            sorted_candidates = sorted(entity_candidates,
                                     key=lambda x: x['similarity'], reverse=True)
            seed_entities = [c['entity_id'] for c in sorted_candidates[:2]]

        logger.info(f"Rule-based filtered {len(entity_candidates)} candidates to {len(seed_entities)} seed entities")
        return seed_entities

    def get_entity_data(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity data from graph store"""
        results = self._execute_graph_query("get_entity_data", {'entity_id': entity_id})
        if results:
            result = results[0]
            return {
                'entity_id': entity_id,
                'name': result['name'],
                'type': result['type'],
                'attributes': json.loads(result['attributes']) if result['attributes'] else {}
            }
        return None

    def get_entity_neighbors(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get entity neighbors with relation types and edge metadata."""

        results = self._execute_graph_query("get_entity_neighbors", {'entity_id': entity_id})
        neighbors: List[Dict[str, Any]] = []

        for result in results:
            neighbor_id = result.get('neighbor_id')
            relation_type = result.get('relation_type')
            if not neighbor_id or not relation_type:
                continue
            mention_count = result.get('mention_count', 0)
            similarity_raw = result.get("similarity")
            try:
                similarity = float(similarity_raw) if similarity_raw is not None else None
            except Exception:
                similarity = None

            # Compute edge weight based on relation type and entity importance
            edge_weight = self.compute_edge_weight(relation_type, mention_count, similarity=similarity)
            neighbors.append(
                {
                    "neighbor_id": neighbor_id,
                    "relation_type": relation_type,
                    "mention_count": mention_count,
                    "similarity": similarity,
                    "edge_weight": edge_weight,
                }
            )

        return neighbors

    def compute_edge_weight(self, relation_type: str, mention_count: int, *, similarity: float | None = None) -> float:
        """Compute edge weight based on relation type and entity importance."""
        # Base weight by relation type
        relation_weights = {
            'MENTIONS': 0.5,  # Chunk-entity relation (lower weight)
            'RELATED_TO': 1.0,
            'PART_OF': 1.2,
            'INSTANCE_OF': 1.1,
            'SIMILAR_TO': 0.9,
            'CONTAINS': 1.2,
            'DEPENDS_ON': 1.3
        }

        base_weight = relation_weights.get(relation_type, 1.0)
        if relation_type == "SIMILAR_TO" and similarity is not None:
            base_weight *= max(0.0, min(float(similarity), 1.0))

        # Boost based on entity importance (mention frequency)
        importance_boost = min(math.log(mention_count + 1) / 10, 0.3)

        return base_weight + importance_boost

    def compute_candidate_priority(self, entity_id: str, hop_distance: int) -> float:
        """Compute priority score for beam search candidate"""
        # Enhanced priority computation with multiple factors

        # Base score: inverse hop distance
        distance_score = 1.0 / (hop_distance + 1)

        # Get entity specificity (degree centrality)
        specificity_score = self.get_entity_specificity(entity_id)

        # Get entity importance (mention count)
        importance_score = self.get_entity_importance(entity_id)

        # Combine scores
        priority = (
            0.4 * distance_score +
            0.3 * specificity_score +
            0.3 * importance_score
        )

        return priority

    def get_entity_specificity(self, entity_id: str) -> float:
        """Get entity specificity based on degree centrality"""
        results = self._execute_graph_query("get_entity_degree", {'entity_id': entity_id})
        degree = results[0]['degree'] if results else 0

        # Normalize degree to [0, 1] range (log scale)
        return min(math.log(degree + 1) / 10, 1.0)

    def get_entity_importance(self, entity_id: str) -> float:
        """Get entity importance based on mention frequency"""
        results = self._execute_graph_query("get_entity_mentions", {'entity_id': entity_id})
        mention_count = results[0]['mention_count'] if results else 0

        # Normalize mention count to [0, 1] range (log scale)
        return min(math.log(mention_count + 1) / 10, 1.0)

    def compute_personalized_pagerank(self, subgraph: SubgraphResult, entity_candidates: List[Dict[str, Any]]) -> PPRResult:
        """Step 3: Personalized PageRank."""
        return _compute_personalized_pagerank(
            config=self.config,
            subgraph=subgraph,
            entity_candidates=entity_candidates,
        )

    def compute_chunk_scores(self, query: str, ppr_result: PPRResult,
                           chunk_candidates: List[Dict[str, Any]]) -> List[ChunkScore]:
        """
        Step 4: Chunk Backtracking and Graph Scoring

        Compute comprehensive scores for chunks based on PPR results
        """
        chunk_scores = []

        # Get top K entities for coverage calculation
        top_entities = sorted(ppr_result.normalized_scores.items(),
                            key=lambda x: x[1], reverse=True)[:self.config.top_k_entities]
        top_entity_ids = {entity_id for entity_id, _ in top_entities}

        # Also get entity backtracking from high PPR entities
        backtracked_chunks = self._backtrack_chunks_from_entities(ppr_result)

        # Combine original candidates with backtracked chunks
        all_chunk_candidates = self._merge_chunk_candidates(chunk_candidates, backtracked_chunks)

        # Process each chunk candidate
        for chunk_candidate in all_chunk_candidates:
            chunk_id = chunk_candidate['chunk_id']
            content = chunk_candidate['content']
            embedding_score = chunk_candidate['similarity']

            # Get entities mentioned in this chunk
            mentioned_entities = self._get_chunk_mentioned_entities(chunk_id)

            # Compute enhanced chunk score with multiple factors
            chunk_score_data = self._compute_enhanced_chunk_score(
                query, chunk_id, content, mentioned_entities, ppr_result,
                top_entity_ids, embedding_score
            )

            chunk_scores.append(ChunkScore(
                chunk_id=chunk_id,
                content=content,
                graph_score=chunk_score_data['graph_score'],
                embedding_score=embedding_score,
                final_score=chunk_score_data['final_score'],
                mentioned_entities=mentioned_entities
            ))

        return chunk_scores

    def _backtrack_chunks_from_entities(self, ppr_result: PPRResult) -> List[Dict[str, Any]]:
        """Backtrack chunks from high PPR entities"""
        # Get top entities by PPR score
        top_entities = sorted(ppr_result.normalized_scores.items(),
                            key=lambda x: x[1], reverse=True)[:self.config.top_k_entities]

        backtracked_chunks = []

        for entity_id, ppr_score in top_entities:
            # Get chunks mentioning this entity
            results = self._execute_graph_query("backtrack_chunks", {
                'entity_id': entity_id,
                'chunks_per_entity': self.config.chunks_per_entity
            })

            for result in results:
                # Compute similarity score (placeholder - would use actual query embedding)
                similarity = 0.5  # Default similarity for backtracked chunks

                backtracked_chunks.append({
                    'chunk_id': result['chunk_id'],
                    'content': result['content'],
                    'similarity': similarity,
                    'source': 'backtracked',
                    'source_entity': entity_id,
                    'ppr_score': ppr_score
                })

        return backtracked_chunks

    def _merge_chunk_candidates(self, original_candidates: List[Dict[str, Any]],
                              backtracked_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge original candidates with backtracked chunks, removing duplicates"""
        seen_chunks = set()
        merged_candidates = []

        # Add original candidates first (higher priority)
        for candidate in original_candidates:
            chunk_id = candidate['chunk_id']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                candidate['source'] = 'semantic_search'
                merged_candidates.append(candidate)

        # Add backtracked chunks if not already present
        for candidate in backtracked_chunks:
            chunk_id = candidate['chunk_id']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                merged_candidates.append(candidate)

        return merged_candidates

    def _get_chunk_mentioned_entities(self, chunk_id: str) -> List[str]:
        """Get entities mentioned in a chunk"""
        results = self._execute_graph_query("get_chunk_entities", {'chunk_id': chunk_id})
        return [result['entity_id'] for result in results]

    def _compute_enhanced_chunk_score(self, query: str, chunk_id: str, content: str,
                                    mentioned_entities: List[str], ppr_result: PPRResult,
                                    top_entity_ids: set, embedding_score: float) -> Dict[str, float]:
        """Compute enhanced chunk score with multiple factors"""

        # 1. Base graph score
        base_graph_score = self._compute_graph_score(
            chunk_id, mentioned_entities, ppr_result, top_entity_ids
        )

        # 2. Mention count score (μ1 factor)
        mention_count_score = self._compute_mention_count_score(mentioned_entities, ppr_result)

        # 3. TF-IDF score (μ2 factor) - simplified implementation
        tfidf_score = self._compute_tfidf_score(query, content, mentioned_entities)

        # 4. Embedding similarity score (μ3 factor)
        embedding_sim_score = embedding_score

        # Combine chunk scoring factors according to design document
        chunk_score = (
            self.config.mu1 * mention_count_score +
            self.config.mu2 * tfidf_score +
            self.config.mu3 * embedding_sim_score
        )

        # Apply coverage boost to base graph score
        coverage = self._compute_coverage(mentioned_entities, top_entity_ids)
        enhanced_graph_score = base_graph_score * (1 + self.config.eta * coverage)

        # Final score fusion
        final_score = (
            self.config.alpha * enhanced_graph_score +
            self.config.beta * embedding_score
        )

        return {
            'graph_score': enhanced_graph_score,
            'chunk_score': chunk_score,
            'final_score': final_score,
            'mention_count_score': mention_count_score,
            'tfidf_score': tfidf_score,
            'coverage': coverage
        }

    def _compute_mention_count_score(self, mentioned_entities: List[str],
                                   ppr_result: PPRResult) -> float:
        """Compute mention count score for chunk"""
        if not mentioned_entities:
            return 0.0

        # Weight mention count by entity PPR scores
        total_score = 0.0
        for entity_id in mentioned_entities:
            ppr_score = ppr_result.normalized_scores.get(entity_id, 0.0)
            # Simple mention count (could be enhanced with actual frequency)
            mention_count = 1.0  # Simplified - in practice would count actual mentions
            total_score += mention_count * ppr_score

        return total_score / len(mentioned_entities) if mentioned_entities else 0.0

    def _compute_tfidf_score(self, query: str, content: str,
                           mentioned_entities: List[str]) -> float:
        """Compute simplified TF-IDF score"""
        if not content or not mentioned_entities:
            return 0.0

        # Simplified TF-IDF implementation
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        # Term frequency in content
        tf_score = len(query_terms & content_terms) / len(content_terms) if content_terms else 0.0

        # Entity relevance boost
        entity_boost = len(mentioned_entities) / 10.0  # Normalize by typical entity count

        return min(tf_score + entity_boost, 1.0)  # Cap at 1.0

    def _compute_graph_score(self, chunk_id: str, mentioned_entities: List[str],
                           ppr_result: PPRResult, top_entity_ids: set) -> float:
        """Compute graph score for a chunk"""
        if not mentioned_entities:
            return 0.0

        # Compute base graph score using softmax aggregation
        entity_scores = []
        for entity_id in mentioned_entities:
            entity_score = self._compute_entity_score(entity_id, ppr_result)
            entity_scores.append(entity_score)

        if not entity_scores:
            return 0.0

        # Softmax aggregation
        exp_scores = [math.exp(score) for score in entity_scores]
        sum_exp = sum(exp_scores)

        if sum_exp == 0:
            base_score = 0.0
        else:
            base_score = sum(
                (exp_score / sum_exp) * score
                for exp_score, score in zip(exp_scores, entity_scores)
            )

        # Compute coverage factor
        coverage = self._compute_coverage(mentioned_entities, top_entity_ids)

        # Apply coverage boost
        graph_score = base_score * (1 + self.config.eta * coverage)

        return graph_score

    def _compute_entity_score(self, entity_id: str, ppr_result: PPRResult) -> float:
        """Compute comprehensive entity score"""
        # Path score (simplified - in production would compute actual shortest paths)
        path_score = self._compute_path_score(entity_id)

        # PPR score
        ppr_score = ppr_result.normalized_scores.get(entity_id, 0.0)

        # Combine scores
        entity_score = (
            self.config.lambda1 * path_score +
            self.config.lambda2 * ppr_score
        )

        return entity_score

    def _compute_path_score(self, entity_id: str) -> float:
        """Compute path score for entity (simplified implementation)"""
        # This is a simplified implementation
        # In production, would compute actual shortest paths from seed entities

        # For now, return a default score based on entity degree
        results = self._execute_graph_query("get_entity_degree", {'entity_id': entity_id})
        degree = results[0]['degree'] if results else 0

        # Simple scoring based on degree (higher degree = higher specificity)
        specificity = math.log(degree + 1)

        # Combine factors (simplified)
        path_score = (
            self.config.gamma1 * 1.0 +  # Assume short path
            self.config.gamma2 * 1.0 +  # Assume high edge weight
            self.config.gamma3 * specificity
        )

        return path_score

    def _compute_coverage(self, mentioned_entities: List[str], top_entity_ids: set) -> float:
        """Compute coverage factor"""
        if not mentioned_entities:
            return 0.0

        intersection = set(mentioned_entities) & top_entity_ids
        coverage = len(intersection) / len(mentioned_entities)

        return coverage

    def fusion_and_ranking(self, chunk_scores: List[ChunkScore]) -> List[ChunkScore]:
        """
        Step 5: Fusion and Ranking

        Final fusion of graph scores and embedding scores with ranking
        """
        if not chunk_scores:
            return []

        # Normalize graph scores
        graph_scores = [cs.graph_score for cs in chunk_scores]
        max_graph_score = max(graph_scores) if graph_scores else 1.0

        # Normalize embedding scores (already normalized from similarity)
        embedding_scores = [cs.embedding_score for cs in chunk_scores]
        max_embedding_score = max(embedding_scores) if embedding_scores else 1.0

        # Update final scores with normalized values
        for chunk_score in chunk_scores:
            normalized_graph_score = chunk_score.graph_score / max_graph_score if max_graph_score > 0 else 0.0
            normalized_embedding_score = chunk_score.embedding_score / max_embedding_score if max_embedding_score > 0 else 0.0

            # Final fusion
            chunk_score.final_score = (
                self.config.alpha * normalized_graph_score +
                self.config.beta * normalized_embedding_score
            )

        # Sort by final score
        chunk_scores.sort(key=lambda x: x.final_score, reverse=True)

        return chunk_scores

    def health_check(self) -> Dict[str, Any]:
        """Health check for the retrieval system"""
        try:
            # Check graph store connection
            graph_health = self.graph_store.health_check()

            # Check embedding model
            test_embedding = self.embedding_model.embed_query("test")
            embedding_health = len(test_embedding) > 0

            return {
                "status": "healthy" if graph_health["status"] == "healthy" and embedding_health else "unhealthy",
                "graph_store": graph_health,
                "embedding_model": embedding_health,
                "config": {
                    "k1_chunks": self.config.k1_chunks,
                    "k2_entities": self.config.k2_entities,
                    "max_hops": self.config.max_hops,
                    "beam_size": self.config.beam_size
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
