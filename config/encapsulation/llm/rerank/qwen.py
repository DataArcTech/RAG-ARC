"""Configuration for Qwen Rerank LLM"""

from framework.config import AbstractConfig
from encapsulation.llm.rerank.qwen import QwenRerankLLM
from typing import Literal, Optional


class QwenRerankConfig(AbstractConfig):
    """Configuration for Qwen Rerank LLM"""
    # Discriminator for config type identification
    type: Literal["qwen_rerank"] = "qwen_rerank"

    # Loading method configuration - can choose between providers
    loading_method: Literal["openai", "huggingface"] = "huggingface"  # Provider for model loading

    # Model configuration
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"  # Path to Qwen reranker model
    device: str = "cuda:0"  # Device for model inference (cuda:0, cuda:1, cpu)
    cache_folder: Optional[str] = None  # Local cache directory for model files

    use_china_mirror: bool = False  # Whether to use domestic mirror source

    # Template configuration for Qwen conversation format
    prefix: str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"  # System prompt template
    suffix: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  # Assistant response template
    instruction: str = "Given the user query, retrieve the relevant passages"  # Default instruction for reranking

    def build(self) -> QwenRerankLLM:
        return QwenRerankLLM(self)