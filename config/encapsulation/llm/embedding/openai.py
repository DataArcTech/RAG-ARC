"""Configuration for OpenAI Embedding LLM"""

import os
from framework.config import AbstractConfig
from encapsulation.llm.embedding.openai import OpenAIEmbeddingLLM
from typing import Literal, Optional


def _resolve_embedding_provider():
    provider = os.getenv("EMBEDDING_MODEL_PROVIDER", "openai").lower()
    return provider if provider in {"openai", "huggingface"} else "openai"


class OpenAIEmbeddingConfig(AbstractConfig):
    """Configuration for OpenAI Embedding LLM"""
    # Discriminator for config type identification
    type: Literal["openai_embedding"] = "openai_embedding"

    # Loading method configuration - can choose between providers
    loading_method: Literal["openai", "huggingface"] = _resolve_embedding_provider()  # Provider for model loading

    # Model configuration
    model_name: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # Embedding model identifier
    embedding_dimensions: Optional[int] = None  # Custom embedding dimensions (only for supported models like text-embedding-3-*)

    # API configuration - loaded from environment variables
    openai_api_key: str = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))  # Embedding provider API key
    openai_base_url: str = os.getenv("EMBEDDING_API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))  # Embedding endpoint URL
    organization: Optional[str] = None  # OpenAI organization ID (optional)

    # Connection configuration
    timeout: float = 60.0  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on failure

    def build(self) -> OpenAIEmbeddingLLM:
        return OpenAIEmbeddingLLM(self)
