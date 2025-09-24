"""Configuration for OpenAI Embedding LLM"""

from framework.config import AbstractConfig
from encapsulation.llm.embedding.openai import OpenAIEmbeddingLLM
from typing import Literal, Optional


class OpenAIEmbeddingConfig(AbstractConfig):
    """Configuration for OpenAI Embedding LLM"""
    # Discriminator for config type identification
    type: Literal["openai_embedding"] = "openai_embedding"

    # Model configuration
    model_name: str = "text-embedding-3-small"  # OpenAI embedding model (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
    embedding_dimensions: Optional[int] = None  # Custom embedding dimensions (only for supported models like text-embedding-3-*)

    # API configuration
    base_url: str = "https://api.gptsapi.net/v1"  # API endpoint URL
    api_key: str = "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2"  # API key for authentication
    organization: Optional[str] = None  # OpenAI organization ID (optional)

    # Connection configuration
    timeout: float = 60.0  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on failure

    def build(self) -> OpenAIEmbeddingLLM:
        return OpenAIEmbeddingLLM(self)