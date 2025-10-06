from abc import abstractmethod
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import logging

from framework.module import AbstractModule

if TYPE_CHECKING:
    from encapsulation.data_model.schema import Chunk

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
        chunks: List['Chunk'],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Chunk reranking

        Args:
            query: Query text
            chunks: List of Chunk objects
            top_k: Return top k results

        Returns:
            List of (chunk_index, score) tuples sorted by score
        """
        pass

    # ==================== UTILITY METHODS ====================
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass