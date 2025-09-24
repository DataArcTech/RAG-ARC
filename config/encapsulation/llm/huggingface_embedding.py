"""Configuration for HuggingFace Embedding LLM"""

from framework.config import AbstractConfig
from encapsulation.llm.embedding.huggingface import HuggingFaceEmbeddingLLM
from typing import Literal, Optional, Dict, Any


class HuggingFaceEmbeddingConfig(AbstractConfig):
    """Configuration for HuggingFace Embedding LLM"""
    # Discriminator for config type identification
    type: Literal["huggingface_embedding"] = "huggingface_embedding"

    # Model configuration
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"  # Path to local model or HuggingFace model ID
    device: str = "cuda:0"  # Device for model inference (cuda:0, cuda:1, cpu)
    cache_folder: Optional[str] = None  # Local cache directory for model files

    # Advanced configuration (optional)
    model_kwargs: Dict[str, Any] = {}  # Additional arguments passed to SentenceTransformer initialization
    encode_kwargs: Dict[str, Any] = {}  # Additional arguments passed to model.encode() method

    use_china_mirror: bool = False  # Whether to use China mirror for HuggingFace
    multi_process: bool = False # Whether to use multi-process for encoding


    def build(self) -> HuggingFaceEmbeddingLLM:
        return HuggingFaceEmbeddingLLM(self)