"""Configuration for OpenAI Embedding LLM"""

import os
from typing import Any, Dict, Literal, Optional

from pydantic import Field
from pydantic import model_validator

from framework.config import AbstractConfig
from encapsulation.llm.embedding.openai import OpenAIEmbeddingLLM


def _resolve_embedding_provider():
    provider = os.getenv("EMBEDDING_MODEL_PROVIDER", "openai").lower()
    return provider if provider in {"openai", "huggingface"} else "openai"


def _default_embedding_model_name() -> str:
    return os.getenv(
        "EMBEDDING_MODEL_NAME",
        os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def _default_embedding_api_key() -> str:
    return os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))


def _default_embedding_base_url() -> str:
    return os.getenv("EMBEDDING_API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))

def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return int(raw)

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class OpenAIEmbeddingConfig(AbstractConfig):
    """Configuration for OpenAI Embedding LLM"""
    # Discriminator for config type identification
    type: Literal["openai_embedding"] = "openai_embedding"

    # Loading method configuration - can choose between providers
    loading_method: Literal["openai", "huggingface"] = Field(default_factory=_resolve_embedding_provider)

    # Model configuration
    model_name: str = Field(default_factory=_default_embedding_model_name)
    embedding_dimensions: Optional[int] = Field(
        default_factory=lambda: _env_int("EMBEDDING_DIMENSIONS")
    )  # Optional override (and required when loading_method="huggingface")

    # HuggingFace-only knobs (used when loading_method="huggingface")
    device: str = Field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", os.getenv("DEVICE", "cpu")))
    cache_folder: Optional[str] = Field(default_factory=lambda: os.getenv("EMBEDDING_CACHE_FOLDER"))
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # API configuration - loaded from environment variables
    openai_api_key: str = Field(default_factory=_default_embedding_api_key)  # Embedding provider API key
    openai_base_url: str = Field(default_factory=_default_embedding_base_url)  # Embedding endpoint URL
    organization: Optional[str] = None  # OpenAI organization ID (optional)

    # Connection configuration
    timeout: float = 60.0  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on failure

    # Request shaping / compatibility
    request_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_REQUEST_BATCH_SIZE", "64")),
        description="Number of inputs per embeddings request when batching is enabled",
    )
    supports_batch_input: bool = Field(
        default_factory=lambda: _env_bool("EMBEDDING_SUPPORTS_BATCH_INPUT", True),
        description="Whether the embeddings endpoint accepts list inputs for the `input` field",
    )

    @model_validator(mode="after")
    def _validate_embedding_dimensions(self):
        # `embedding_dimensions` is optional:
        # - If provided, it is used as an override.
        # - If missing, EmbeddingLLMBase can auto-detect the dimension by running a tiny embedding call.
        return self

    def build(self) -> OpenAIEmbeddingLLM:
        return OpenAIEmbeddingLLM(self)
