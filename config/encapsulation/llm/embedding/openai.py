"""Configuration for OpenAI Embedding LLM"""

from framework.config import AbstractConfig
from encapsulation.llm.embedding.openai import OpenAIEmbeddingLLM
from typing import Literal, Optional


class OpenAIEmbeddingConfig(AbstractConfig):
    """Configuration for OpenAI Embedding LLM"""
    # Discriminator for config type identification
    type: Literal["openai_embedding"] = "openai_embedding"

    # Loading method configuration - can choose between providers
    loading_method: Literal["openai", "huggingface"] = "openai"  # Provider for model loading

    # Model configuration
    model_name: str = "text-embedding-3-small"  # OpenAI embedding model (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
    embedding_dimensions: Optional[int] = None  # Custom embedding dimensions (only for supported models like text-embedding-3-*)

    # API configuration - loaded from environment variables
    openai_api_key: str # API key for authentication
    openai_base_url: str # API endpoint URL (optional, defaults to OpenAI)
    organization: Optional[str] = None  # OpenAI organization ID (optional)

    # Connection configuration
    timeout: float = 60.0  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on failure

    def build(self) -> OpenAIEmbeddingLLM:
        return OpenAIEmbeddingLLM(self)