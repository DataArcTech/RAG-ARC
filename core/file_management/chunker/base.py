from abc import abstractmethod
from typing import List, Optional, Dict, Any

from framework.module import AbstractModule


class AbstractChunker(AbstractModule):
    """
    Abstract base class for text chunking strategies.

    This class defines the interface for different chunking strategies that can split
    text into manageable chunks for downstream processing in the RAG pipeline.
    """

    @abstractmethod
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks using this chunking strategy.

        Args:
            text: The input text to be chunked
            metadata: Optional metadata about the source content
            **kwargs: Strategy-specific chunking options

        Returns:
            List of chunk dictionaries, each containing:
            - content: The chunk text content
            - metadata: Chunk-specific metadata (start_idx, end_idx, chunk_id, etc.)
            - source_metadata: Original source metadata (if provided)

        Raises:
            ValueError: If text is invalid or chunking parameters are invalid
            Exception: If chunking fails
        """
        pass

    @abstractmethod
    def get_chunker_info(self) -> Dict[str, Any]:
        """
        Get information about this chunker's configuration and capabilities.

        Returns:
            Dictionary containing:
            - strategy: The chunking strategy type (e.g., "fixed_size", "semantic", "recursive")
            - chunk_size: Maximum chunk size (if applicable)
            - chunk_overlap: Overlap between chunks (if applicable)
            - supported_features: List of supported features
            - parameters: Current configuration parameters

        """
        pass