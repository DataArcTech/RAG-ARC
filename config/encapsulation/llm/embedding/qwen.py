"""Configuration for Qwen Embedding LLM"""

from framework.config import AbstractConfig
from encapsulation.llm.embedding.qwen import QwenEmbeddingLLM
from typing import Literal, Optional, Dict, Any


class QwenEmbeddingConfig(AbstractConfig):
    """Configuration for Qwen Embedding LLM"""
    # Discriminator for config type identification
    type: Literal["qwen_embedding"] = "qwen_embedding"

    # Loading method configuration - can choose between providers
    loading_method: Literal["openai", "huggingface"] = "huggingface"  # Provider for model loading

    # Model configuration
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"  # Path to local model or HuggingFace model ID
    device: str = "cuda:0"  # Device for model inference (cuda:0, cuda:1, cpu)
    cache_folder: Optional[str] = None  # Local cache directory for model files

    use_china_mirror: bool = False  # Whether to use domestic mirror source

    # Advanced configuration (optional)
    model_kwargs: Dict[str, Any] = {}  # Additional arguments passed to SentenceTransformer initialization
    encode_kwargs: Dict[str, Any] = {}  # Additional arguments passed to model.encode() method

    def build(self) -> QwenEmbeddingLLM:
        return QwenEmbeddingLLM(self)