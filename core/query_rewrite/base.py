from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    List,
    Dict,
)
import logging

from framework.module import AbstractModule


class AbstractQueryRewriter(AbstractModule):
    """
    Abstract base class for query rewriting strategies in RAG systems.

    Query rewriting is a critical component in RAG pipelines that transforms user queries
    to improve retrieval effectiveness. This base class defines the interface that all
    query rewriting implementations must follow.

    RAG Pipeline Position:
        User Query → Query Rewrite → Retrieval → Rerank → LLM Generate Answer
                     ↑ This component
    """

    @abstractmethod
    def rewrite_query(
        self,
        query: str,
        instruction: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Rewrite a query to improve retrieval effectiveness.

        Primary configuration (instruction, temperature, max_tokens) comes from config
        during initialization. Instruction parameter allows runtime override.

        Args:
            query: Original user query to rewrite
            instruction: Optional override for config instruction template
            **kwargs: Additional strategy-specific parameters

        Returns:
            Rewritten query string

        Raises:
            ValueError: If query is empty or invalid
            Exception: If rewriting process fails
        """
        pass

    @abstractmethod
    def get_rewriter_info(self) -> Dict[str, Any]:
        """
        Get information about this query rewriter's capabilities and configuration.

        Returns:
            Dictionary containing rewriter information
        """
        pass