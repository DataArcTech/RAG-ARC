from framework.config import AbstractConfig
from config.encapsulation.llm.rerank.qwen import QwenRerankConfig
from config.encapsulation.llm.rerank.listwise import ListwiseRerankConfig
from core.rerank.llm_reranker import LLMReranker
from typing import Literal, Annotated, Union
from pydantic import Field

class LLMRerankerConfig(AbstractConfig):
    """
    Configuration for LLM-based Reranker.

    This config accepts any RerankLLMBase implementation through dependency injection,
    making it flexible to use with different reranker providers (Qwen, BGE, Cohere, etc.).
    """
    type: Literal["llm_reranker"] = "llm_reranker"
    rerank_llm_config: Annotated[Union[QwenRerankConfig, ListwiseRerankConfig], Field(discriminator="type")]

    def build(self) -> LLMReranker:
        return LLMReranker(self)
