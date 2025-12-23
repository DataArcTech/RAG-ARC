from abc import abstractmethod
from typing import Dict, Any, List, Union, Optional
import logging

from framework.module import AbstractModule

logger = logging.getLogger(__name__)


class EmbeddingLLMBase(AbstractModule):
    """
    Base class for embedding LLM implementations
    Supports text vectorization with sync/async capabilities
    """

    # ==================== EMBEDDING METHODS ====================
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate text embeddings"""
        pass

    # ==================== ASYNC EMBEDDING METHODS ====================
    @abstractmethod
    async def aembed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate text embeddings asynchronously"""
        pass

    # ==================== CONVENIENCE METHODS ====================
    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple chunks - always returns list of embeddings"""
        result = self.embed(texts)
        if isinstance(texts, str):
            return [result] if isinstance(result[0], (int, float)) else result
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed single query - always returns single embedding"""
        result = self.embed(text)
        return result if isinstance(result, list) and isinstance(result[0], (int, float)) else result[0]

    # ==================== UTILITY METHODS ====================
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

    def get_embedding_dimension(self) -> int:
        """
        Return the embedding vector dimension for the current embedding model.

        Resolution order:
        1) Configured dimension (`config.embedding_dimensions`) when provided.
        2) Lazy auto-detect by issuing a single embedding request and caching the result.
        """
        configured = getattr(self.config, "embedding_dimensions", None)
        if configured is not None:
            return int(configured)

        cached = getattr(self, "_embedding_dimension_cache", None)
        if cached is not None:
            return int(cached)

        probe = self.embed_query("dimension_probe")
        if not isinstance(probe, list) or not probe:
            raise RuntimeError("Failed to detect embedding dimension: empty probe embedding returned")

        detected = len(probe)
        setattr(self, "_embedding_dimension_cache", detected)
        return detected
