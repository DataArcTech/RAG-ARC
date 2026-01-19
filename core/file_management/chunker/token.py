from typing import List, Optional, Dict, Any, TYPE_CHECKING
from bisect import bisect_left, bisect_right
import logging
import re

from .base import AbstractChunker
from core.file_management.atomic_units.markdown_atomic import split_markdown_atomic_blocks

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

    _URL_RE = re.compile(r"https?://[^\s<>\[\]{}()\"']+", re.IGNORECASE)
    _URL_TRAILING_PUNCT = ".,;:!?)]}\"'"

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
        url_atomic_context_tokens = kwargs.get(
            'url_atomic_context_tokens',
            getattr(self.config, 'url_atomic_context_tokens', 10),
        )

        try:
            chunk_units: List[tuple[str, str]] = []
            for kind, segment in split_markdown_atomic_blocks(text):
                if kind in {"code", "table", "image"}:
                    chunk_units.append((kind, segment))
                    continue
                for piece in self._split_on_tokens(
                    segment,
                    chunk_size,
                    chunk_overlap,
                    url_atomic_context_tokens,
                ):
                    if piece:
                        chunk_units.append((kind, piece))

            # Convert to standardized format
            result = []
            current_pos = 0

            for i, (segment_type, chunk_content) in enumerate(chunk_units):
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
                        'strategy': 'token',
                        'segment_type': segment_type,
                        'is_fenced_code_block': segment_type == "code",
                        'is_atomic_block': segment_type in {"code", "table", "image"},
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
            'supported_features': [
                'token_awareness',
                'overlap_control',
                'model_compatibility',
                'fenced_code_atomic',
                'url_atomic',
            ],
            'parameters': {
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'encoding_name': encoding_name,
                'model_name': model_name,
                'url_atomic_context_tokens': getattr(self.config, 'url_atomic_context_tokens', 10),
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

    def _split_on_tokens(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        url_atomic_context_tokens: int,
    ) -> List[str]:
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
        url_atomic_spans = self._build_url_atomic_spans(
            text=text,
            tokenizer=tokenizer,
            input_ids=input_ids,
            context_tokens=max(0, int(url_atomic_context_tokens or 0)),
        )
        span_starts = [start for start, _ in url_atomic_spans]
        span_ends = [end for _, end in url_atomic_spans]

        def _decode_utf8_without_replacement(
            token_ids: List[int],
        ) -> str:
            chunk_bytes = tokenizer.decode_bytes(token_ids)
            try:
                return chunk_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # If the token slice cuts through a multi-byte UTF-8 sequence, decoding the
                # raw bytes fails. Fall back to ignoring invalid edge bytes to avoid U+FFFD.
                return chunk_bytes.decode("utf-8", errors="ignore")

        def _decode_token_window(start: int, end: int) -> tuple[str, int, int]:
            # Try to expand/shrink token boundaries slightly so the decoded bytes form valid UTF-8.
            # This avoids introducing U+FFFD (replacement chars) that can appear when using
            # tokenizer.decode() on token slices cutting through multi-byte sequences.
            max_adjust = 4
            n = len(input_ids)
            for start_shift in range(0, max_adjust + 1):
                s = max(0, start - start_shift)
                for end_shift in range(0, max_adjust + 1):
                    e = min(n, end + end_shift)
                    if s >= e:
                        continue
                    chunk_bytes = tokenizer.decode_bytes(input_ids[s:e])
                    try:
                        return chunk_bytes.decode("utf-8"), s, e
                    except UnicodeDecodeError:
                        continue

            # Last resort: decode with invalid bytes ignored (still guarantees no U+FFFD).
            return _decode_utf8_without_replacement(input_ids[start:end]), start, end

        def _adjust_end_for_atomic_spans(boundary: int) -> int:
            if not url_atomic_spans:
                return boundary
            idx = bisect_left(span_starts, boundary) - 1
            if idx >= 0 and span_starts[idx] < boundary <= span_ends[idx]:
                return min(len(input_ids), span_ends[idx] + 1)
            return boundary

        def _adjust_start_for_atomic_spans(boundary: int) -> int:
            if not url_atomic_spans:
                return boundary
            idx = bisect_left(span_starts, boundary) - 1
            if idx >= 0 and span_starts[idx] < boundary <= span_ends[idx]:
                return span_starts[idx]
            return boundary

        def _adjust_start_forward_for_atomic_spans(boundary: int) -> int:
            if not url_atomic_spans:
                return boundary
            idx = bisect_left(span_starts, boundary) - 1
            if idx >= 0 and span_starts[idx] < boundary <= span_ends[idx]:
                return min(len(input_ids), span_ends[idx] + 1)
            return boundary

        # Split tokens into chunks with overlap
        splits = []
        start_idx = 0

        while start_idx < len(input_ids):
            end_idx = min(start_idx + chunk_size, len(input_ids))
            end_idx = _adjust_end_for_atomic_spans(end_idx)
            chunk_text, _, actual_end = _decode_token_window(start_idx, end_idx)
            adjusted_end = _adjust_end_for_atomic_spans(actual_end)
            if adjusted_end != actual_end:
                chunk_text, _, actual_end = _decode_token_window(start_idx, adjusted_end)
            splits.append(chunk_text)

            if actual_end >= len(input_ids):
                break

            next_start = max(0, actual_end - chunk_overlap)
            next_start = _adjust_start_for_atomic_spans(next_start)
            if next_start <= start_idx:
                next_start = start_idx + max(1, chunk_size - chunk_overlap)
                next_start = _adjust_start_forward_for_atomic_spans(next_start)
            start_idx = next_start

        return splits

    def _build_url_atomic_spans(
        self,
        text: str,
        tokenizer: Any,
        input_ids: List[int],
        context_tokens: int,
    ) -> List[tuple[int, int]]:
        if not text or not input_ids:
            return []
        url_spans = self._find_url_spans(text)
        if not url_spans:
            return []
        token_starts, token_ends = self._token_byte_offsets(tokenizer, input_ids)
        spans: List[tuple[int, int]] = []
        for start_char, end_char in url_spans:
            byte_start = len(text[:start_char].encode("utf-8"))
            byte_end = len(text[:end_char].encode("utf-8"))
            start_token = bisect_right(token_ends, byte_start)
            end_token = bisect_left(token_starts, byte_end) - 1
            if start_token < 0 or end_token < 0 or start_token > end_token:
                continue
            span_start = max(0, start_token - context_tokens)
            span_end = min(len(input_ids) - 1, end_token + context_tokens)
            spans.append((span_start, span_end))
        return self._merge_spans(spans)

    def _find_url_spans(self, text: str) -> List[tuple[int, int]]:
        spans: List[tuple[int, int]] = []
        for match in self._URL_RE.finditer(text):
            start, end = match.span()
            while end > start and text[end - 1] in self._URL_TRAILING_PUNCT:
                end -= 1
            if end > start:
                spans.append((start, end))
        return spans

    def _token_byte_offsets(self, tokenizer: Any, input_ids: List[int]) -> tuple[List[int], List[int]]:
        starts: List[int] = []
        ends: List[int] = []
        offset = 0
        decode_single = getattr(tokenizer, "decode_single_token_bytes", None)
        for token_id in input_ids:
            if decode_single:
                token_bytes = decode_single(token_id)
            else:
                token_bytes = tokenizer.decode_bytes([token_id])
            starts.append(offset)
            offset += len(token_bytes)
            ends.append(offset)
        return starts, ends

    @staticmethod
    def _merge_spans(spans: List[tuple[int, int]]) -> List[tuple[int, int]]:
        if not spans:
            return []
        spans.sort()
        merged = [list(spans[0])]
        for start, end in spans[1:]:
            last = merged[-1]
            if start <= last[1] + 1:
                last[1] = max(last[1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]
