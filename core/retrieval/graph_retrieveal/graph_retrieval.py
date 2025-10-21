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

from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from pydantic import Field
import logging
import json
import numpy as np
import math

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_retrieveal.base import BaseGraphRetriever

if TYPE_CHECKING:
    from config.core.retrieval.graph_retrieval_config import GraphRetrievalConfig

logger = logging.getLogger(__name__)



@dataclass
class CandidateResult:
    """Candidate recall result"""
    entity_candidates: List[Dict[str, Any]] = field(default_factory=list)  # [(entity_id, entity_name, similarity)]
    chunk_candidates: List[Dict[str, Any]] = field(default_factory=list)   # [(chunk_id, content, similarity)]


@dataclass
class SubgraphResult:
    """Subgraph construction result"""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # entity_id -> entity_data
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # [(from_id, relation_type, to_id)]
    seed_entities: List[str] = field(default_factory=list)


@dataclass
class PPRResult:
    """Personalized PageRank result"""
    scores: Dict[str, float] = field(default_factory=dict)  # entity_id -> ppr_score
    normalized_scores: Dict[str, float] = field(default_factory=dict)  # entity_id -> normalized_score


@dataclass
class ChunkScore:
    """Chunk scoring result"""
    chunk_id: str
    content: str
    graph_score: float
    embedding_score: float
    final_score: float
    mentioned_entities: List[str] = field(default_factory=list)





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
        if self.graph_store_type.startswith("neo4j"):
            # Use Cypher queries for Neo4j
            return self._execute_neo4j_query(query_type, params)
        elif self.graph_store_type.startswith("networkx"):
            # Use NetworkX-specific queries
            return self._execute_networkx_query(query_type, params)
        else:
            raise ValueError(f"Unsupported graph store type: {self.graph_store_type}")

    def _execute_neo4j_query(self, query_type: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute Neo4j Cypher queries"""
        params = params or {}

        if query_type == "semantic_search_entities":
            query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Chunk)
            WITH e, count(d) as mention_count
            RETURN e.id_ as entity_id,
                   e.entity_name as entity_name,
                   e.entity_type as entity_type,
                   e.embedding as embedding,
                   mention_count,
                   e.attributes as attributes
            ORDER BY mention_count DESC
            LIMIT 2000
            """
        elif query_type == "semantic_search_chunks":
            query = """
            MATCH (d:Chunk)
            WHERE d.embedding IS NOT NULL
            OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
            WITH d, count(e) as entity_count, collect(e.entity_name) as entity_names
            RETURN d.id_ as chunk_id,
                   d.content as content,
                   d.embedding as embedding,
                   d.metadata as metadata,
                   entity_count,
                   entity_names
            ORDER BY entity_count DESC
            LIMIT 2000
            """
        elif query_type == "get_entity_data":
            query = """
            MATCH (e:Entity {id_: $entity_id})
            RETURN e.entity_name as name, e.entity_type as type, e.attributes as attributes
            """
        elif query_type == "get_entity_neighbors":
            query = """
            MATCH (e1:Entity {id_: $entity_id})-[r]->(e2:Entity)
            OPTIONAL MATCH (e2)<-[:MENTIONS]-(d:Chunk)
            WITH e2, type(r) as relation_type, count(d) as mention_count
            RETURN e2.id_ as neighbor_id, relation_type, mention_count
            UNION
            MATCH (e1:Entity)-[r]->(e2:Entity {id_: $entity_id})
            OPTIONAL MATCH (e1)<-[:MENTIONS]-(d:Chunk)
            WITH e1, type(r) as relation_type, count(d) as mention_count
            RETURN e1.id_ as neighbor_id, relation_type, mention_count
            """
        elif query_type == "get_entity_degree":
            query = """
            MATCH (e:Entity {id_: $entity_id})-[r]-()
            RETURN count(r) as degree
            """
        elif query_type == "get_entity_mentions":
            query = """
            MATCH (e:Entity {id_: $entity_id})<-[:MENTIONS]-(d:Chunk)
            RETURN count(d) as mention_count
            """
        elif query_type == "get_chunk_entities":
            query = """
            MATCH (d:Chunk {id_: $chunk_id})-[:MENTIONS]->(e:Entity)
            RETURN e.id_ as entity_id
            """
        elif query_type == "backtrack_chunks":
            query = """
            MATCH (e:Entity {id_: $entity_id})<-[:MENTIONS]-(d:Chunk)
            WHERE d.embedding IS NOT NULL
            RETURN d.id_ as chunk_id, d.content as content, d.embedding as embedding
            LIMIT $chunks_per_entity
            """
        else:
            raise ValueError(f"Unknown query type: {query_type}")

        return self.graph_store.query(query, params)

    def _execute_networkx_query(self, query_type: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute NetworkX-specific queries"""
        params = params or {}
        results = []

        if query_type == "semantic_search_entities":
            # Get all entity nodes with embeddings
            for node_id, node_data in self.graph_store.graph.nodes(data=True):
                if (node_data.get('node_type') == 'Entity' and
                    node_data.get('embedding')):

                    # Count mentions (chunks that mention this entity)
                    mention_count = 0
                    for edge in self.graph_store.graph.edges(data=True):
                        source, target, edge_data = edge
                        if (target == node_id and
                            edge_data.get('relation_type') == 'MENTIONS'):
                            mention_count += 1

                    results.append({
                        'entity_id': node_data.get('id_'),
                        'entity_name': node_data.get('entity_name'),
                        'entity_type': node_data.get('entity_type'),
                        'embedding': node_data.get('embedding'),
                        'mention_count': mention_count,
                        'attributes': node_data.get('attributes', '{}')
                    })

            # Sort by mention count
            results.sort(key=lambda x: x['mention_count'], reverse=True)
            return results[:2000]

        elif query_type == "semantic_search_chunks":
            # Get all chunk nodes with embeddings
            for node_id, node_data in self.graph_store.graph.nodes(data=True):
                if (node_data.get('node_type') == 'Chunk' and
                    node_data.get('embedding')):

                    # Count entities and collect entity names
                    entity_count = 0
                    entity_names = []
                    for edge in self.graph_store.graph.edges(data=True):
                        source, target, edge_data = edge
                        if (source == node_id and
                            edge_data.get('relation_type') == 'MENTIONS'):
                            entity_count += 1
                            target_data = self.graph_store.graph.nodes[target]
                            if target_data.get('entity_name'):
                                entity_names.append(target_data['entity_name'])

                    results.append({
                        'chunk_id': node_data.get('id_'),
                        'content': node_data.get('content'),
                        'embedding': node_data.get('embedding'),
                        'metadata': node_data.get('metadata', '{}'),
                        'entity_count': entity_count,
                        'entity_names': entity_names
                    })

            # Sort by entity count
            results.sort(key=lambda x: x['entity_count'], reverse=True)
            return results[:2000]

        elif query_type == "get_entity_data":
            entity_id = params.get('entity_id')
            entity_node_id = f"entity_{entity_id}"

            if self.graph_store.graph.has_node(entity_node_id):
                node_data = self.graph_store.graph.nodes[entity_node_id]
                return [{
                    'name': node_data.get('entity_name'),
                    'type': node_data.get('entity_type'),
                    'attributes': node_data.get('attributes', '{}')
                }]
            return []

        elif query_type == "get_entity_neighbors":
            entity_id = params.get('entity_id')
            entity_node_id = f"entity_{entity_id}"

            if not self.graph_store.graph.has_node(entity_node_id):
                return []

            neighbors = []
            # Get outgoing edges
            for target in self.graph_store.graph.successors(entity_node_id):
                edge_data = self.graph_store.graph.get_edge_data(entity_node_id, target)
                if edge_data:
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('relation_type') != 'MENTIONS':
                            # Count mentions for target entity
                            mention_count = 0
                            for edge in self.graph_store.graph.edges(data=True):
                                if (edge[1] == target and
                                    edge[2].get('relation_type') == 'MENTIONS'):
                                    mention_count += 1

                            target_data = self.graph_store.graph.nodes[target]
                            neighbors.append({
                                'neighbor_id': target_data.get('id_'),
                                'relation_type': edge_attrs.get('relation_type'),
                                'mention_count': mention_count
                            })

            # Get incoming edges
            for source in self.graph_store.graph.predecessors(entity_node_id):
                edge_data = self.graph_store.graph.get_edge_data(source, entity_node_id)
                if edge_data:
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('relation_type') != 'MENTIONS':
                            # Count mentions for source entity
                            mention_count = 0
                            for edge in self.graph_store.graph.edges(data=True):
                                if (edge[1] == source and
                                    edge[2].get('relation_type') == 'MENTIONS'):
                                    mention_count += 1

                            source_data = self.graph_store.graph.nodes[source]
                            neighbors.append({
                                'neighbor_id': source_data.get('id_'),
                                'relation_type': edge_attrs.get('relation_type'),
                                'mention_count': mention_count
                            })

            return neighbors

        elif query_type == "get_entity_degree":
            entity_id = params.get('entity_id')
            entity_node_id = f"entity_{entity_id}"

            if self.graph_store.graph.has_node(entity_node_id):
                degree = self.graph_store.graph.degree(entity_node_id)
                return [{'degree': degree}]
            return [{'degree': 0}]

        elif query_type == "get_entity_mentions":
            entity_id = params.get('entity_id')
            entity_node_id = f"entity_{entity_id}"

            mention_count = 0
            for edge in self.graph_store.graph.edges(data=True):
                if (edge[1] == entity_node_id and
                    edge[2].get('relation_type') == 'MENTIONS'):
                    mention_count += 1

            return [{'mention_count': mention_count}]

        elif query_type == "get_chunk_entities":
            chunk_id = params.get('chunk_id')
            chunk_node_id = f"chunk_{chunk_id}"

            entity_ids = []
            for target in self.graph_store.graph.successors(chunk_node_id):
                edge_data = self.graph_store.graph.get_edge_data(chunk_node_id, target)
                if edge_data:
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('relation_type') == 'MENTIONS':
                            target_data = self.graph_store.graph.nodes[target]
                            entity_ids.append({'entity_id': target_data.get('id_')})

            return entity_ids

        elif query_type == "backtrack_chunks":
            entity_id = params.get('entity_id')
            entity_node_id = f"entity_{entity_id}"
            chunks_per_entity = params.get('chunks_per_entity', 10)

            chunks = []
            for source in self.graph_store.graph.predecessors(entity_node_id):
                edge_data = self.graph_store.graph.get_edge_data(source, entity_node_id)
                if edge_data:
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('relation_type') == 'MENTIONS':
                            source_data = self.graph_store.graph.nodes[source]
                            if (source_data.get('node_type') == 'Chunk' and
                                source_data.get('embedding')):
                                chunks.append({
                                    'chunk_id': source_data.get('id_'),
                                    'content': source_data.get('content'),
                                    'embedding': source_data.get('embedding')
                                })

                                if len(chunks) >= chunks_per_entity:
                                    break

                    if len(chunks) >= chunks_per_entity:
                        break

            return chunks

        else:
            raise ValueError(f"Unknown query type: {query_type}")

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
                        'mentioned_entities': chunk_score.mentioned_entities
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
                            'mentioned_entities': chunk_score.mentioned_entities
                        }
                    )
                    chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Could not retrieve chunk {chunk_score.chunk_id}: {e}")
                # Create a fallback chunk
                chunk = Chunk(
                    id=chunk_score.chunk_id,
                    content=chunk_score.content,
                    metadata={
                        'score': chunk_score.final_score,
                        'graph_score': chunk_score.graph_score,
                        'embedding_score': chunk_score.embedding_score,
                        'mentioned_entities': chunk_score.mentioned_entities
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
                popularity_boost = min(math.log(mention_count + 1) / 10, 0.1)

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

                # Add edges and collect next level candidates
                for neighbor_id, relation_type, edge_weight in neighbors:
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
                        'index': i,
                        'entity_id': candidate['entity_id'],
                        'entity_name': candidate.get('entity_name', 'Unknown'),
                        'entity_type': candidate.get('entity_type', 'Entity'),
                        'attributes': attr_text
                    })

                # Create consolidated prompt for LLM
                prompt = f"""Given the user query: "{query}"

Please select the most relevant entities from the following candidates that would be useful as seed entities for graph-based retrieval. Consider:
1. Semantic relevance to the query
2. Entity type and specificity
3. Entity attributes and their relevance
4. Potential to lead to relevant information through graph traversal

Entity candidates:
"""

                for entity in entity_info:
                    prompt += f"""
{entity['index']}: {entity['entity_name']}
   - Type: {entity['entity_type']}
   - Attributes: {entity['attributes']}"""

                prompt += """

Please respond with only the indices (numbers) of the selected entities, separated by commas.
Select 2-5 most relevant entities. For example: 0,2,4

Selected indices:"""

                # Get LLM response
                messages = [{"role": "user", "content": prompt}]
                response = self.llm.chat(messages)

                # Parse LLM response to get selected entity indices
                try:
                    import re
                    numbers = re.findall(r'\d+', response.strip())

                    # Convert to integers and filter valid indices
                    selected_indices = []
                    for num_str in numbers:
                        idx = int(num_str)
                        if 0 <= idx < len(entity_candidates):
                            selected_indices.append(idx)

                    # Remove duplicates while preserving order
                    seen = set()
                    unique_indices = []
                    for idx in selected_indices:
                        if idx not in seen:
                            seen.add(idx)
                            unique_indices.append(idx)

                    # Ensure we have at least some entities (fallback to top 2 if parsing fails)
                    if not unique_indices:
                        logger.warning("Failed to parse LLM response, using top 2 entities as fallback")
                        unique_indices = [0, 1] if len(entity_candidates) >= 2 else [0] if len(entity_candidates) >= 1 else []

                    # Limit to reasonable number of seed entities
                    selected_indices = unique_indices[:5]

                except Exception as e:
                    logger.warning(f"Error parsing LLM response: {e}. Using top 2 entities as fallback.")
                    selected_indices = [0, 1] if len(entity_candidates) >= 2 else [0] if len(entity_candidates) >= 1 else []

                # Convert indices to entity IDs
                seed_entities = [entity_candidates[i]['entity_id'] for i in selected_indices]

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

    def get_entity_neighbors(self, entity_id: str) -> List[Tuple[str, str, float]]:
        """Get entity neighbors with relation types and edge weights"""
        # Enhanced neighbor query with additional metadata for better edge weighting
        results = self._execute_graph_query("get_entity_neighbors", {'entity_id': entity_id})
        neighbors = []

        for result in results:
            neighbor_id = result['neighbor_id']
            relation_type = result['relation_type']
            mention_count = result.get('mention_count', 0)

            # Compute edge weight based on relation type and entity importance
            edge_weight = self.compute_edge_weight(relation_type, mention_count)
            neighbors.append((neighbor_id, relation_type, edge_weight))

        return neighbors

    def compute_edge_weight(self, relation_type: str, mention_count: int) -> float:
        """Compute edge weight based on relation type and entity importance"""
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

    def compute_personalized_pagerank(self, subgraph: SubgraphResult,
                                    entity_candidates: List[Dict[str, Any]]) -> PPRResult:
        """
        Step 3: Personalized PageRank

        Compute PPR scores for entities in the subgraph
        """
        if not subgraph.nodes:
            return PPRResult()

        # Build adjacency matrix
        entity_ids = list(subgraph.nodes.keys())
        n = len(entity_ids)
        id_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids)}

        # Initialize weighted adjacency matrix (undirected graph)
        adj_matrix = np.zeros((n, n))
        edge_weights = {}  # Store edge weights for better transition matrix

        for from_id, relation_type, to_id in subgraph.edges:
            if from_id in id_to_idx and to_id in id_to_idx:
                i, j = id_to_idx[from_id], id_to_idx[to_id]

                # Get edge weight (could be computed based on relation type)
                weight = self._get_ppr_edge_weight(relation_type)

                adj_matrix[i][j] = weight
                adj_matrix[j][i] = weight  # Undirected
                edge_weights[(i, j)] = weight
                edge_weights[(j, i)] = weight

        # Normalize adjacency matrix (transition matrix)
        transition_matrix = np.zeros((n, n))
        for i in range(n):
            row_sum = np.sum(adj_matrix[i])
            if row_sum > 0:
                transition_matrix[i] = adj_matrix[i] / row_sum
            else:
                # Handle isolated nodes: uniform transition to all nodes
                transition_matrix[i] = np.ones(n) / n

        # Compute personalization vector
        personalization_vector = self._compute_personalization_vector(
            entity_ids, entity_candidates, subgraph
        )

        # Run PPR algorithm
        ppr_scores = self._run_ppr_algorithm(
            transition_matrix, personalization_vector
        )

        # Convert back to entity_id -> score mapping
        scores = {}
        for i, entity_id in enumerate(entity_ids):
            scores[entity_id] = ppr_scores[i]

        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        normalized_scores = {entity_id: score / max_score for entity_id, score in scores.items()}

        return PPRResult(scores=scores, normalized_scores=normalized_scores)

    def _compute_personalization_vector(self, entity_ids: List[str],
                                      entity_candidates: List[Dict[str, Any]],
                                      subgraph: SubgraphResult) -> np.ndarray:
        """Compute personalization vector for PPR"""
        n = len(entity_ids)
        personalization = np.zeros(n)

        # Create mapping for quick lookup
        id_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids)}
        candidate_similarities = {c['entity_id']: c['similarity'] for c in entity_candidates}

        # Compute personalization scores
        total_score = 0.0
        for i, entity_id in enumerate(entity_ids):
            sim_e = candidate_similarities.get(entity_id, 0.0)
            triple_boost = self._compute_triple_boost(entity_id, subgraph)

            score = self.config.beta1 * sim_e + self.config.beta2 * triple_boost
            personalization[i] = score
            total_score += score

        # Normalize to probability distribution
        if total_score > 0:
            personalization = personalization / total_score
        else:
            # Uniform distribution if no scores
            personalization = np.ones(n) / n

        return personalization

    def _compute_triple_boost(self, entity_id: str, subgraph: SubgraphResult) -> float:
        """Compute triple boost score for entity"""
        # Simple implementation: count of relations
        # In production, could be more sophisticated
        relation_count = 0
        for from_id, _, to_id in subgraph.edges:
            if from_id == entity_id or to_id == entity_id:
                relation_count += 1

        # Normalize by log to avoid extreme values
        return math.log(relation_count + 1)

    def _run_ppr_algorithm(self, transition_matrix: np.ndarray,
                          personalization_vector: np.ndarray) -> np.ndarray:
        """Run the PPR algorithm"""
        n = len(personalization_vector)

        # Initialize PPR scores
        ppr_scores = personalization_vector.copy()

        # Iterative computation
        for iteration in range(self.config.max_iterations):
            old_scores = ppr_scores.copy()

            # PPR update: (1-d) * personalization + d * transition * scores
            ppr_scores = (
                (1 - self.config.damping_factor) * personalization_vector +
                self.config.damping_factor * np.dot(transition_matrix.T, ppr_scores)
            )

            # Check convergence
            diff = np.linalg.norm(ppr_scores - old_scores)
            if diff < self.config.tolerance:
                logger.info(f"PPR converged after {iteration + 1} iterations")
                break

        return ppr_scores

    def _get_ppr_edge_weight(self, relation_type: str) -> float:
        """Get edge weight for PPR computation based on relation type"""
        # Configure relation weights for PPR (different from subgraph construction)
        ppr_weights = {
            'MENTIONS': 0.3,  # Lower weight for document mentions
            'RELATED_TO': 1.0,
            'PART_OF': 1.5,   # Higher weight for hierarchical relations
            'INSTANCE_OF': 1.3,
            'SIMILAR_TO': 0.8,
            'CONTAINS': 1.4,
            'DEPENDS_ON': 1.2,
            'SYNONYM': 0.6    # Lower weight for synonym relations
        }

        return ppr_weights.get(relation_type, 1.0)

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
