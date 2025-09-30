from framework.config import AbstractConfig
from config.encapsulation.llm.rerank.qwen import QwenRerankConfig
from core.rerank.llm_reranker import LLMReranker
from typing import Literal, Union

class LLMRerankerConfig(AbstractConfig):
    """
    Configuration for LLM-based Reranker.

    This config accepts any RerankLLMBase implementation through dependency injection,
    making it flexible to use with different reranker providers (Qwen, BGE, Cohere, etc.).
    """
    type: Literal["llm_reranker"] = "llm_reranker"
    rerank_llm_config: QwenRerankConfig  # Accept any RerankLLM config

    def build(self) -> LLMReranker:
        return LLMReranker(self)