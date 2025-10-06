from typing import Literal

from framework.config import AbstractConfig
from application.rag_inference.module import RAGInference
from config.core.query_rewrite_config import LLMQueryRewriterConfig
from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
from config.core.rerank_config import LLMRerankerConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig


class RAGInferenceConfig(AbstractConfig):
    type: Literal["rag_inference"] = "rag_inference"
    query_rewrite_config: LLMQueryRewriterConfig
    retrieval_config: MultiPathRetrieverConfig
    reranker_config: LLMRerankerConfig
    llm_config: OpenAIChatConfig
    
    def build(self):
        return RAGInference(self)