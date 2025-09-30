from typing import Dict, Any, List, TYPE_CHECKING
from .base import AbstractReranker
from encapsulation.data_model.schema import Document

import logging

if TYPE_CHECKING:
    from config.core.rerank_config import LLMRerankerConfig

logger = logging.getLogger(__name__)


class LLMReranker(AbstractReranker):
    """
    LLM-based document reranker for RAG systems.

    Uses any reranking LLM to score and reorder documents based on relevance to the query.
    This implementation is LLM-agnostic and works with any RerankLLMBase implementation
    from the encapsulation layer (Qwen, BGE, Cohere, etc.).

    RAG Pipeline Position:
        User Query → Query Rewrite → Retrieval → Rerank → LLM Generate Answer
                                                   ↑ This component

    Key features:
    - LLM-agnostic: Works with any RerankLLMBase implementation
    - Configurable batch size for memory optimization
    - Top-k filtering with default value from config
    - Score normalization and metadata enrichment
    - Integration with RAG-ARC LLM infrastructure via dependency injection
    """

    def __init__(self, config):
        super().__init__(config)
        # Build rerank LLM from sub-config following framework pattern
        # Accepts any RerankLLMBase implementation (Qwen, BGE, Cohere, etc.)
        self.rerank_llm = config.rerank_llm_config.build()

    def rerank(
        self,
        query: str,
        documents: List[Document],
        **kwargs: Any
    ) -> List[Document]:
        """
        Rerank documents using any reranking LLM.

        All configuration is handled by the encapsulation layer. Core layer
        focuses on document structure and metadata management.

        Args:
            query: User query to rank documents against
            documents: List of Document objects from retrieval step
            **kwargs: Parameters passed through to encapsulation layer

        Returns:
            List of Document objects reordered by relevance, with rerank scores
            added to metadata

        Raises:
            ValueError: If query is empty or documents list is invalid
            RuntimeError: If reranking process fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            logger.warning("Empty documents list provided to reranker")
            return []

        try:
            # Pass all parameters to encapsulation layer
            # Encapsulation layer handles all configuration (top_k, batch_size, instruction)
            ranked_results = self.rerank_llm.rerank(
                query=query,
                documents=documents,
                **kwargs  # Pass through all parameters to encapsulation layer
            )

            # Convert results back to Document objects with metadata
            # ranked_results is List[Tuple[int, float]] from encapsulation layer
            reranked_documents = []
            for doc_idx, score in ranked_results:
                doc = documents[doc_idx]
                # Preserve original metadata and add rerank score
                new_metadata = doc.metadata.copy()
                new_metadata["rerank_score"] = score
                new_metadata["rerank_method"] = self.rerank_llm.get_model_info().get("provider", "llm")

                # Create new Document with updated metadata
                reranked_doc = Document(
                    content=doc.content,
                    metadata=new_metadata,
                    id=doc.id
                )
                reranked_documents.append(reranked_doc)

            logger.info(f"Reranked {len(documents)} documents, returned {len(reranked_documents)}")
            return reranked_documents

        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            # Return original documents as fallback
            logger.warning("Using original document order as fallback")
            return documents

    def get_reranker_info(self) -> Dict[str, Any]:
        """
        Get information about this reranker's configuration.

        Returns:
            Dictionary containing reranker information
        """
        return {
            "type": "llm_reranker",
            "llm_info": self.rerank_llm.get_model_info(),
            "supports_batch_processing": True,
            "fallback_strategy": "original_order"
        }