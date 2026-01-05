import json
import logging
import uuid
from typing import List, Optional, Union

import numpy as np

from encapsulation.data_model.schema import Chunk
from core.utils.owner_guard import is_admin_owner, normalize_owner_id

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jRetrieveMixin:
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
            top_k_facts, top_k_fact_indices = self._rerank_facts(query, query_fact_scores, fact_ids, owner_id=owner_filter)
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
                owner_id=owner_filter,
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
            owner_id=owner_filter,
        )

        # Step 7: Convert to Chunk objects
        chunks = self._convert_to_chunks(sorted_doc_ids[:top_k], sorted_doc_scores[:top_k], owner_id=owner_filter)

        # Optionally attach subgraph information for visualization
        if return_subgraph_info and chunks:
            node_to_ppr_score = ppr_scores_dict  # Already a dict in Neo4j version

            subgraph_info = {
                "subgraph_nodes": list(subgraph_nodes),
                "seed_entity_ids": list(seed_entity_ids),
                "retrieved_chunk_ids": sorted_doc_ids[:top_k],
                "node_ppr_scores": node_to_ppr_score,
                "query": query,
            }
            if chunks[0].metadata is None:
                chunks[0].metadata = {}
            chunks[0].metadata["_subgraph_info"] = subgraph_info

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
            results = self.graph_store._execute_query(
                query,
                {
                    "chunk_ids": chunk_ids,
                    "owner_id": str(owner_id),
                },
            )
        else:
            query = """
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunk_ids
            RETURN c.chunk_id AS chunk_id, c.content AS content, c.owner_id AS owner_id, c.metadata AS metadata
            """
            results = self.graph_store._execute_query(query, {"chunk_ids": chunk_ids})

        # Build chunk_id -> chunk data mapping
        chunk_data_map = {}
        for record in results:
            chunk_id = record["chunk_id"]
            owner_value = self.graph_store._restore_owner_id(record.get("owner_id"))
            chunk_data_map[chunk_id] = {
                "content": record["content"],
                "owner_id": owner_value,
                "metadata": record["metadata"],
            }

        # Create Chunk objects in the same order as chunk_ids
        chunks = []
        for chunk_id, score in zip(chunk_ids, scores):
            if chunk_id in chunk_data_map:
                data = chunk_data_map[chunk_id]

                # Parse metadata
                try:
                    metadata = json.loads(data["metadata"]) if data["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Add score to metadata
                metadata["score"] = float(score)

                # Create Chunk object with restored owner type
                owner_value = data.get("owner_id")
                owner_field: Optional[Union[str, uuid.UUID]] = None
                if owner_value:
                    try:
                        owner_field = uuid.UUID(owner_value)
                    except (ValueError, TypeError):
                        owner_field = owner_value

                chunk = Chunk(
                    id=chunk_id,
                    content=data["content"],
                    owner_id=owner_field,
                    metadata=metadata,
                )
                chunks.append(chunk)

        return chunks

