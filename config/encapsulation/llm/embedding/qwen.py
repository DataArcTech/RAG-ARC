"""Configuration for Qwen Embedding LLM"""

import os
from framework.config import AbstractConfig
from encapsulation.llm.embedding.qwen import QwenEmbeddingLLM
from typing import Literal, Optional, Dict, Any

from pydantic import Field, model_validator


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


class QwenEmbeddingConfig(AbstractConfig):
    """Configuration for Qwen Embedding LLM"""
    # Discriminator for config type identification
    type: Literal["qwen_embedding"] = "qwen_embedding"

    # Loading method configuration - Qwen embedding is a local SentenceTransformer in this project
    loading_method: Literal["huggingface"] = "huggingface"

    # Model configuration
    model_name: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"))
    device: str = Field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", os.getenv("DEVICE", "cpu")))
    cache_folder: Optional[str] = Field(default_factory=lambda: os.getenv("EMBEDDING_CACHE_FOLDER"))
    embedding_dimensions: Optional[int] = Field(default_factory=lambda: _env_int("EMBEDDING_DIMENSIONS"))

    use_china_mirror: bool = Field(default_factory=lambda: _env_bool("USE_CHINA_MIRROR", False))

    # Advanced configuration (optional)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)  # Additional arguments passed to SentenceTransformer initialization
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)  # Additional arguments passed to model.encode() method

    @model_validator(mode="after")
    def _validate_embedding_dimensions(self):
        if self.loading_method == "huggingface" and self.embedding_dimensions is None:
            raise ValueError(
                "embedding_dimensions is required for local HuggingFace embeddings. "
                "Set EMBEDDING_DIMENSIONS (or provide embedding_dimensions in config JSON)."
            )
        return self

    def build(self) -> QwenEmbeddingLLM:
        return QwenEmbeddingLLM(self)
