from framework.config import AbstractConfig
from config.encapsulation.qwen_config import QwenLLMConfig
from core.rerank.qwen3 import Qwen3Reranker
from typing import Literal

class Qwen3RerankerConfig(AbstractConfig):
    """Configuration for Qwen3 Reranker"""
    type: Literal["qwen_reranker"] = "qwen_reranker"
    qwen3_llm_config: QwenLLMConfig

    def build(self) -> Qwen3Reranker:
        return Qwen3Reranker(self)