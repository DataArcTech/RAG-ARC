"""Configuration classes for all chunker strategies"""

from framework.config import AbstractConfig
from core.file_management.chunker.token import TokenChunker
from core.file_management.chunker.recursive import RecursiveChunker
from core.file_management.chunker.markdown_header import MarkdownHeaderChunker
from core.file_management.chunker.semantic import SemanticChunker
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from typing import Literal, Optional, List, Union


class TokenChunkerConfig(AbstractConfig):
    """Configuration for Token Chunker - token-aware text chunking using tiktoken"""
    # Discriminator for config type identification
    type: Literal["token_chunker"] = "token_chunker"

    # Core chunking parameters
    chunk_size: int = 1000  # Maximum token count per chunk (default: 4000 in implementation)
    chunk_overlap: int = 100  # Overlap token count between chunks (default: 200 in implementation)

    # Tokenizer configuration
    encoding_name: str = "gpt2"  # Encoder name for tiktoken (default: 'gpt2')
    model_name: Optional[str] = None  # Model name for model-specific tokenizer (optional)

    # Advanced tokenizer settings (commented out as they use defaults)
    # allowed_special: set = set()  # Set of allowed special tokens (default: empty set)
    # disallowed_special: str = "all"  # Disallowed special tokens handling (default: "all")

    def build(self) -> TokenChunker:
        return TokenChunker(self)


class RecursiveChunkerConfig(AbstractConfig):
    """Configuration for Recursive Chunker - hierarchical text splitting with separator priorities"""
    # Discriminator for config type identification
    type: Literal["recursive_chunker"] = "recursive_chunker"

    # Core chunking parameters
    chunk_size: int = 500  # Maximum character count per chunk (default: 200 in implementation)
    chunk_overlap: int = 50  # Overlap character count between chunks (default: 0 in implementation)

    # Separator configuration (commented out as they use defaults)
    # separators: List[str] = ["\n\n", "\n", ". ", " ", ""]  # Separator hierarchy (default: ["\n\n", "\n", "#"])
    # keep_separator: Union[bool, Literal["start", "end"]] = True  # Separator retention strategy (default: True)
    # is_separator_regex: bool = False  # Whether separators are regex patterns (default: False)

    def build(self) -> RecursiveChunker:
        return RecursiveChunker(self)


class MarkdownHeaderChunkerConfig(AbstractConfig):
    """Configuration for Markdown Header Chunker - structure-aware markdown splitting"""
    # Discriminator for config type identification
    type: Literal["markdown_header_chunker"] = "markdown_header_chunker"

    # Header configuration (commented out as they use defaults)
    # headers_to_split_on: List[str] = ["#", "##", "###"]  # Header markers to split on (default: ["#", "##"])
    strip_headers: bool = False  # Whether to remove headers from chunk content (default: True)
    # chunk_size: int = 0  # Maximum chunk size for sub-splitting, 0 disables (default: 0)

    def build(self) -> MarkdownHeaderChunker:
        return MarkdownHeaderChunker(self)


class SemanticChunkerConfig(AbstractConfig):
    """Configuration for Semantic Chunker - embedding-based semantic coherence chunking"""
    # Discriminator for config type identification
    type: Literal["semantic_chunker"] = "semantic_chunker"

    # Required embedding configuration
    embedding: QwenEmbeddingConfig  # Embedding model config for semantic analysis

    # Semantic analysis parameters (commented out as they use defaults)
    # buffer_size: int = 1  # Context window size around each sentence (default: 1)
    # breakpoint_threshold_type: str = "percentile"  # Statistical method for breakpoints (default: 'percentile')
    # breakpoint_threshold_amount: Optional[float] = 95  # Threshold value (default: None, uses method defaults)
    # number_of_chunks: Optional[int] = None  # Fixed number of chunks to create (default: None)
    # sentence_split_regex: str = r"(?<=[.?!])\s+"  # Pattern for sentence splitting (default: same)
    # min_chunk_size: Optional[int] = 50  # Minimum chunk size filter (default: None)
    # add_start_index: bool = False  # Whether to track character positions (default: False)

    def build(self) -> SemanticChunker:
        return SemanticChunker(self)