from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    List,
    Dict,
)

from framework.module import AbstractModule
from encapsulation.data_model.schema import Chunk


class AbstractReranker(AbstractModule):
    """
    Abstract base class for chunks reranking in RAG systems.

    Reranking is a critical component that reorders retrieved chunks based on
    their relevance to the user query, improving the quality of context provided
    to the generation model.

    RAG Pipeline Position:
        User Query → Query Rewrite → Retrieval → Rerank → LLM Generate Answer
                                                   ↑ This component
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List["Chunk"],
        **kwargs: Any
    ) -> List["Chunk"]:
        """
        Rerank chunks based on relevance to the query.

        All configuration parameters are handled by the encapsulation layer.
        Core layer focuses on chunk structure and metadata management.

        Args:
            query: User query to rank chunks against
            chunks: List of Chunk objects from retrieval step
            **kwargs: Parameters passed through to encapsulation layer

        Returns:
            List of Chunk objects reordered by relevance with rerank scores
            in metadata

        Raises:
            ValueError: If query is empty or chunks list is invalid
            Exception: If reranking process fails
        """
        pass

    @abstractmethod
    def get_reranker_info(self) -> Dict[str, Any]:
        """
        Get information about this reranker's capabilities and configuration.

        Returns:
            Dictionary containing reranker information
        """
        pass