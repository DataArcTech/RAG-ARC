from typing import List, Optional, Dict, Any, Union, Literal, TYPE_CHECKING
import re
import logging

from .base import AbstractChunker

if TYPE_CHECKING:
    from config.core.file_management.chunker.chunker_config import RecursiveChunkerConfig

logger = logging.getLogger(__name__)


class RecursiveChunker(AbstractChunker):
    """
    RecursiveChunker is a text chunker that recursively splits text using multiple separators in priority order.

    This class implements hierarchical text splitting using different separators (paragraphs, sentences, etc.) in a structured manner,
    preserving natural text boundaries where possible while maintaining specified chunk size constraints.

    Key features:
    - Hierarchical splitting with multiple separator priorities
    - Natural text boundary preservation (paragraphs, sentences, words)
    - Support for both literal and regex separators
    - Configurable separator retention strategies
    - Intelligent fallback to character-level splitting when necessary
    - Overlap mechanism to maintain context between chunks
    - Validates chunk size constraints to prevent infinite recursion

    Main parameters (from config):
        chunk_size (int): Maximum character count per chunk, defaults to 200
        chunk_overlap (int): Overlap character count between chunks, defaults to 0
        separators (List[str]): List of separators in priority order, defaults to ["\n\n", "\n", "#"]
        keep_separator (Union[bool, Literal["start", "end"]]): How to handle separators, defaults to True
        is_separator_regex (bool): Whether separators are regex patterns, defaults to False

    Core methods:
        - chunk_text: Main chunking method using recursive splitting strategy
        - get_chunker_info: Get chunker configuration information
        - _split_recursive: Execute recursive splitting logic with separator hierarchy
        - _chunk_text: Force split text into fixed-size chunks with overlap

    Performance considerations:
        - Prioritizes natural boundaries over strict size limits for better readability
        - Recursive approach may have higher computational cost for deeply nested structures
        - Overlap mechanism maintains context but increases total content size
        - For very large texts, consider using simpler chunking strategies
        - Regex separators provide flexibility but may impact performance

    Typical usage:
        >>> config = RecursiveChunkerConfig(chunk_size=500, separators=["\n\n", "\n", ". "])
        >>> chunker = RecursiveChunker(config=config)
        >>> chunks = chunker.chunk_text("long text...")
        >>> info = chunker.get_chunker_info()

    Separator strategies:
        - Paragraphs first: ["\n\n", "\n", ". ", " ", ""]
        - Sentences first: [". ", "! ", "? ", "\n", " ", ""]
        - Custom regex: [r"\n\n", r"\n", r"[.!?]\s+"] with is_separator_regex=True
    """

    def __init__(self, config: "RecursiveChunkerConfig"):
        """Initialize RecursiveChunker with config"""
        super().__init__(config)

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Split text using recursive character-based chunking.

        This method attempts to split text using a hierarchy of separators, starting with the most
        preferred separators (like paragraphs) and falling back to less preferred ones (like spaces)
        when chunks are still too large. This approach preserves natural text structure while
        respecting size constraints.

        Args:
            text: Text content to be chunked
            metadata: Optional source document metadata, will be passed to each chunk
            **kwargs: Runtime parameter overrides
                chunk_size (int): Maximum characters per chunk, overrides config value
                chunk_overlap (int): Chunk overlap characters, overrides config value
                separators (List[str]): Custom separator list, overrides config value

        Returns:
            List of chunk dictionaries, each containing:
            - content: Text content of the chunk
            - metadata: Chunk-specific metadata
                - chunk_id: Unique identifier for the chunk
                - chunk_index: Index of chunk in sequence
                - start_idx: Starting position in original text
                - end_idx: Ending position in original text
                - character_count: Character count of this chunk
                - strategy: Chunking strategy identifier ('recursive')
            - source_metadata: Original document metadata (if provided)

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap > chunk_size
            Exception: If chunking process fails
        """
        # Get config parameters
        chunk_size = getattr(self.config, 'chunk_size', 200)
        chunk_overlap = getattr(self.config, 'chunk_overlap', 0)
        separators = getattr(self.config, 'separators', ["\n\n", "\n", "#"])
        keep_separator = getattr(self.config, 'keep_separator', True)
        is_separator_regex = getattr(self.config, 'is_separator_regex', False)

        # Allow runtime parameter overrides
        chunk_size = kwargs.get('chunk_size', chunk_size)
        chunk_overlap = kwargs.get('chunk_overlap', chunk_overlap)
        separators = kwargs.get('separators', separators)

        # Validate parameters
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )

        try:
            chunks = self._split_recursive(text, separators, chunk_size, chunk_overlap, keep_separator, is_separator_regex)

            # Convert to standardized format
            result = []
            current_pos = 0

            for i, chunk_content in enumerate(chunks):
                start_idx = text.find(chunk_content, current_pos)
                if start_idx == -1:
                    start_idx = current_pos

                end_idx = start_idx + len(chunk_content)
                current_pos = end_idx

                chunk_dict = {
                    'content': chunk_content,
                    'metadata': {
                        'chunk_id': i,
                        'chunk_index': i,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'character_count': len(chunk_content),
                        'strategy': 'recursive'
                    }
                }

                # Add source metadata if provided
                if metadata:
                    chunk_dict['source_metadata'] = metadata.copy()

                result.append(chunk_dict)

            logger.info(f"Split text into {len(result)} chunks using recursive strategy")
            return result

        except Exception as e:
            logger.error(f"Failed to chunk text with recursive strategy: {str(e)}")
            raise

    def get_chunker_info(self) -> Dict[str, Any]:
        """
        Get information about this chunker's configuration and capabilities.

        Returns:
            Dictionary containing detailed chunker information:
            - strategy: Chunking strategy type ('recursive')
            - chunk_size: Maximum character count limit
            - chunk_overlap: Overlap character count between chunks
            - separators: List of separators in priority order
            - keep_separator: Separator retention strategy
            - is_separator_regex: Whether separators are treated as regex patterns
            - supported_features: List of supported feature capabilities
            - parameters: Complete copy of current configuration parameters
        """
        chunk_size = getattr(self.config, 'chunk_size', 200)
        chunk_overlap = getattr(self.config, 'chunk_overlap', 0)
        separators = getattr(self.config, 'separators', ["\n\n", "\n", "#"])
        keep_separator = getattr(self.config, 'keep_separator', True)
        is_separator_regex = getattr(self.config, 'is_separator_regex', False)

        return {
            'strategy': 'recursive',
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'separators': separators,
            'keep_separator': keep_separator,
            'is_separator_regex': is_separator_regex,
            'supported_features': ['hierarchical_splitting', 'natural_boundaries', 'overlap_control'],
            'parameters': {
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'separators': separators,
                'keep_separator': keep_separator,
                'is_separator_regex': is_separator_regex
            }
        }

    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
        chunk_overlap: int,
        keep_separator: Union[bool, Literal["start", "end"]],
        is_separator_regex: bool
    ) -> List[str]:
        """
        Recursively split text using separator hierarchy.

        This method implements the core recursive splitting logic, trying separators in order
        of preference. If the first separator doesn't adequately reduce chunk sizes, it
        recursively tries the next separator in the list.

        Args:
            text: Text to be split
            separators: List of separators to try in order
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap characters between chunks
            keep_separator: How to handle separators in results
            is_separator_regex: Whether to treat separators as regex patterns

        Returns:
            List of text chunks split according to separator hierarchy
        """
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            return self._chunk_text(text, chunk_size, chunk_overlap)

        sep = separators[0]
        if is_separator_regex:
            parts = re.split(f"({sep})", text)
        else:
            parts = text.split(sep)

        if len(parts) == 1:
            return self._split_recursive(text, separators[1:], chunk_size, chunk_overlap, keep_separator, is_separator_regex)

        chunks, current = [], ""
        for i, p in enumerate(parts):
            if not p:
                continue

            if is_separator_regex:
                is_sep = bool(re.fullmatch(sep, p))
            else:
                is_sep = (p == sep)

            if is_sep:
                if keep_separator is True or keep_separator == "end":
                    current += p
                elif keep_separator == "start":
                    if current:
                        chunks.append(current)
                    current = p
                continue

            if len(current) + len(p) > chunk_size and current:
                chunks.extend(self._split_recursive(current, separators[1:], chunk_size, chunk_overlap, keep_separator, is_separator_regex))
                current = p
            else:
                current += p

        if current:
            chunks.extend(self._split_recursive(current, separators[1:], chunk_size, chunk_overlap, keep_separator, is_separator_regex))

        return chunks

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Force split text into fixed-size chunks with overlap.

        This method is used as a last resort when no separators can adequately split
        the text. It performs character-level chunking with specified overlap.

        Args:
            text: Text to be force-split
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap characters between chunks

        Returns:
            List of fixed-size text chunks with overlap
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap
        return chunks