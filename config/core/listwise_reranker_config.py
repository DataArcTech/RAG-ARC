from framework.config import AbstractConfig
from config.encapsulation.llm.rerank.listwise import ListwiseRerankConfig
from core.rerank.listwise_reranker import ListwiseReranker
from typing import Literal

class ListwiseRerankerConfig(AbstractConfig):
    """
    Configuration for Listwise LLM-based Reranker.

    This config uses chat LLMs for listwise ranking where the model ranks
    all documents at once by outputting a ranked list of indices.
    """
    type: Literal["listwise_reranker"] = "listwise_reranker"
    rerank_llm_config: ListwiseRerankConfig  # Listwise rerank LLM config

    def build(self) -> ListwiseReranker:
        return ListwiseReranker(self)