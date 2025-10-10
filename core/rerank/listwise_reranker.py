from typing import Dict, Any, List, TYPE_CHECKING
from .base import AbstractReranker
from encapsulation.data_model.schema import Chunk

import logging

if TYPE_CHECKING:
    from config.core.rerank_config import ListwiseRerankerConfig

logger = logging.getLogger(__name__)


class ListwiseReranker(AbstractReranker):
    """
    Listwise LLM-based document reranker for RAG systems.

    Uses listwise ranking approach where the LLM ranks all documents at once
    by outputting a ranked list of indices. This implementation is LLM-agnostic
    and works with any chat LLM that supports OpenAI-compatible API.

    RAG Pipeline Position:
        User Query → Query Rewrite → Retrieval → Rerank → LLM Generate Answer
                                                   ↑ This component

    Key features:
    - Listwise ranking: Ranks all documents together for better global ordering
    - Reasoning-based: LLM provides step-by-step reasoning for ranking decisions
    - LLM-agnostic: Works with any OpenAI-compatible chat API
    - Configurable prompts: Supports custom prompt templates
    - Top-k filtering with default value from config
    - Score assignment based on ranking position
    - Integration with RAG-ARC LLM infrastructure via dependency injection

    Differences from LLMReranker:
    - LLMReranker: Uses pointwise/pairwise scoring models (e.g., Qwen reranker)
    - ListwiseReranker: Uses chat LLMs with listwise ranking approach
    """

    def __init__(self, config: "ListwiseRerankerConfig"):
        super().__init__(config)
        # Build listwise rerank LLM from sub-config following framework pattern
        self.rerank_llm = config.rerank_llm_config.build()

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        **kwargs: Any
    ) -> List[Chunk]:
        """
        Rerank chunks using listwise ranking approach.

        All configuration is handled by the encapsulation layer. Core layer
        focuses on chunk structure and metadata management.

        Args:
            query: User query to rank chunks against
            chunks: List of Chunk objects from retrieval step
            **kwargs: Parameters passed through to encapsulation layer

        Returns:
            List of Chunk objects reordered by relevance, with rerank scores
            added to metadata

        Raises:
            ValueError: If query is empty or chunks list is invalid
            RuntimeError: If reranking process fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not chunks:
            logger.warning("Empty chunks list provided to reranker")
            return []

        try:
            # Pass all parameters to encapsulation layer
            # Encapsulation layer handles all configuration (top_k, temperature, prompt_template)
            ranked_results = self.rerank_llm.rerank(
                query=query,
                chunks=chunks,
                **kwargs  # Pass through all parameters to encapsulation layer
            )

            # Convert results back to Chunk objects with metadata
            # ranked_results is List[Tuple[int, float]] from encapsulation layer
            reranked_chunks = []
            for chunk_idx, score in ranked_results:
                chunk = chunks[chunk_idx]
                # Preserve original metadata and add rerank score
                new_metadata = chunk.metadata.copy()
                new_metadata["rerank_score"] = score
                new_metadata["rerank_method"] = "listwise"

                # Create new Chunk with updated metadata
                reranked_chunk = Chunk(
                    content=chunk.content,
                    metadata=new_metadata,
                    id=chunk.id,
                    graph=chunk.graph
                )
                reranked_chunks.append(reranked_chunk)

            logger.info(f"Listwise reranked {len(chunks)} chunks, returned {len(reranked_chunks)}")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Listwise chunk reranking failed: {e}")
            # Return original chunks as fallback
            logger.warning("Using original chunk order as fallback")
            return chunks

    def get_reranker_info(self) -> Dict[str, Any]:
        """
        Get information about this reranker's configuration.

        Returns:
            Dictionary containing reranker information
        """
        return {
            "type": "listwise_reranker",
            "llm_info": self.rerank_llm.get_model_info(),
            "ranking_approach": "listwise",
            "supports_reasoning": True,
            "fallback_strategy": "original_order"
        }

