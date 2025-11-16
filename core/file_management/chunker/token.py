from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging

from .base import AbstractChunker

if TYPE_CHECKING:
    from config.core.file_management.chunker.chunker_config import TokenChunkerConfig

logger = logging.getLogger(__name__)


class TokenChunker(AbstractChunker):
    """
    TokenChunker is a text chunker based on model tokenizers, suitable for scenarios requiring precise token count control.

    This class implements token-aware text chunking by integrating tokenizer libraries like tiktoken, ensuring chunks respect token boundaries
    and stay within specified token limits for LLM processing pipelines.

    Key features:
    - Token-level precise chunking, avoiding word or semantic unit truncation
    - Support for multiple model tokenizers (GPT series, custom models, etc.)
    - Configurable chunk overlap to ensure contextual continuity
    - Automatic token counting and boundary detection
    - Support for special token handling and encoding parameter configuration
    - Compatible with all tiktoken library encoding modes

    Main parameters (from config):
        chunk_size (int): Maximum token count per chunk, defaults to 4000
        chunk_overlap (int): Overlap token count between chunks, defaults to 200
        encoding_name (str): Encoder name, defaults to 'gpt2'
        model_name (str): Model name, optional, used to get model-specific tokenizer
        allowed_special (set): Set of allowed special tokens
        disallowed_special (str|set): Disallowed special tokens, defaults to 'all'

    Core methods:
        - chunk_text: Main chunking method, splits text into token-aware chunks
        - get_chunker_info: Get chunker configuration information
        - _split_on_tokens: Execute token-based text splitting
        - _encode: Encode text to token ID list

    Performance considerations:
        - Token-level chunking ensures LLM compatibility
        - Overlap mechanism maintains semantic continuity but increases total token consumption
        - For large texts, recommend adjusting chunk_size to balance performance and precision
        - tiktoken library provides efficient token encoding performance

    Typical usage:
        >>> config = TokenChunkerConfig(chunk_size=1000, chunk_overlap=100)
        >>> chunker = TokenChunker(config=config)
        >>> chunks = chunker.chunk_text("long text...")
        >>> info = chunker.get_chunker_info()

    Dependencies:
        - tiktoken: For token encoding and decoding
        - Corresponding model tokenizer configuration
    """

    def __init__(self, config: "TokenChunkerConfig"):
        """Initialize TokenChunker with tokenizer configuration"""
        super().__init__(config)

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks using tokenizer-aware splitting.

        This method splits long text into chunks suitable for LLM processing based on specified token limits and overlap settings.
        Each chunk is strictly controlled within token limits and maintains appropriate overlap between chunks to preserve semantic continuity.

        Args:
            text: Text content to be chunked
            metadata: Optional source metadata, will be passed to each chunk
            **kwargs: Runtime parameter overrides
                chunk_size (int): Maximum tokens per chunk, overrides config value
                chunk_overlap (int): Chunk overlap tokens, overrides config value

        Returns:
            List of chunk dictionaries, each containing:
            - content: Text content of the chunk
            - metadata: Chunk-specific metadata
                - chunk_id: Unique identifier for the chunk
                - chunk_index: Index of chunk in sequence
                - start_idx: Starting position in original text
                - end_idx: Ending position in original text
                - token_count: Token count of this chunk
                - strategy: Chunking strategy identifier ('token')
            - source_metadata: Original source metadata (if provided)

        Raises:
            ImportError: If tiktoken library is not installed
            ValueError: If text is empty or parameters are invalid
            Exception: If chunking process fails
        """
        # Get config parameters
        chunk_size = getattr(self.config, 'chunk_size', 4000)
        chunk_overlap = getattr(self.config, 'chunk_overlap', 200)

        # Allow runtime parameter overrides
        chunk_size = kwargs.get('chunk_size', chunk_size)
        chunk_overlap = kwargs.get('chunk_overlap', chunk_overlap)

        try:
            chunks = self._split_on_tokens(text, chunk_size, chunk_overlap)

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
                        'token_count': len(self._encode(chunk_content)),
                        'strategy': 'token'
                    }
                }

                # Add source metadata if provided
                if metadata:
                    chunk_dict['source_metadata'] = metadata.copy()

                result.append(chunk_dict)

            logger.info(f"Split text into {len(result)} chunks using token strategy")
            return result

        except Exception as e:
            logger.error(f"Failed to chunk text with token strategy: {str(e)}")
            raise

    def get_chunker_info(self) -> Dict[str, Any]:
        """
        Get information about this chunker's configuration and capabilities.

        Returns:
            Dictionary containing detailed chunker information:
            - strategy: Chunking strategy type ('token')
            - chunk_size: Maximum token count limit
            - chunk_overlap: Overlap token count between chunks
            - encoding_name: Encoder name being used
            - model_name: Associated model name (if any)
            - supported_features: List of supported feature capabilities
            - parameters: Complete copy of current configuration parameters
        """
        chunk_size = getattr(self.config, 'chunk_size', 4000)
        chunk_overlap = getattr(self.config, 'chunk_overlap', 200)
        encoding_name = getattr(self.config, 'encoding_name', 'gpt2')
        model_name = getattr(self.config, 'model_name', None)

        return {
            'strategy': 'token',
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'encoding_name': encoding_name,
            'model_name': model_name,
            'supported_features': ['token_awareness', 'overlap_control', 'model_compatibility'],
            'parameters': {
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'encoding_name': encoding_name,
                'model_name': model_name
            }
        }

    def _encode(self, text: str) -> List[int]:
        """
        Encode text to token ID list.

        Args:
            text: Text to be encoded

        Returns:
            List of integer token IDs

        Raises:
            ImportError: If tiktoken library is not installed
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed for TokenChunker. "
                "Please install dependencies with: uv sync"
            )

        model_name = getattr(self.config, 'model_name', None)
        encoding_name = getattr(self.config, 'encoding_name', 'gpt2')
        allowed_special = getattr(self.config, 'allowed_special', set())
        disallowed_special = getattr(self.config, 'disallowed_special', "all")

        if model_name is not None:
            tokenizer = tiktoken.encoding_for_model(model_name)
        else:
            tokenizer = tiktoken.get_encoding(encoding_name)

        return tokenizer.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

    def _split_on_tokens(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Core logic for executing text splitting using tokenizer.

        This method first encodes text into token sequence, then splits the token sequence into multiple segments
        according to specified chunk_size and overlap parameters, finally decodes each token segment back to text.

        Args:
            text: Original text to be split
            chunk_size: Maximum token count per chunk
            chunk_overlap: Overlap token count between adjacent chunks

        Returns:
            List of split text chunks

        Raises:
            ImportError: If tiktoken library is not installed
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed for TokenChunker. "
                "Please install dependencies with: uv sync"
            )

        model_name = getattr(self.config, 'model_name', None)
        encoding_name = getattr(self.config, 'encoding_name', 'gpt2')
        allowed_special = getattr(self.config, 'allowed_special', set())
        disallowed_special = getattr(self.config, 'disallowed_special', "all")

        if model_name is not None:
            tokenizer = tiktoken.encoding_for_model(model_name)
        else:
            tokenizer = tiktoken.get_encoding(encoding_name)

        # Encode text to tokens
        input_ids = tokenizer.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

        # Split tokens into chunks with overlap
        splits = []
        start_idx = 0

        while start_idx < len(input_ids):
            end_idx = min(start_idx + chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:end_idx]
            chunk_text = tokenizer.decode(chunk_ids)
            splits.append(chunk_text)

            if end_idx == len(input_ids):
                break

            start_idx += chunk_size - chunk_overlap

        return splits