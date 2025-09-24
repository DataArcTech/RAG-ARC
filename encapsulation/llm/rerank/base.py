from abc import abstractmethod
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import logging

from framework.module import AbstractModule

if TYPE_CHECKING:
    from encapsulation.data_model.schema import Document

logger = logging.getLogger(__name__)


class RerankLLMBase(AbstractModule):
    """
    Base class for reranking LLM implementations
    Supports document relevance scoring and ranking
    """

    # ==================== RERANKING METHODS ====================
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List['Document'],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Document reranking

        Args:
            query: Query text
            documents: List of Document objects
            top_k: Return top k results

        Returns:
            List of (document_index, score) tuples sorted by score
        """
        pass

    # ==================== UTILITY METHODS ====================
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass