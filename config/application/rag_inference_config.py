from typing import Annotated, Literal, Union

from pydantic import Field

from framework.config import AbstractConfig
from application.rag_inference.module import RAGInference
from config.core.query_rewrite_config import LLMQueryRewriterConfig
from config.core.query_rewrite.noop import NoOpQueryRewriterConfig
from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
from config.core.rerank_config import LLMRerankerConfig
from config.core.rerank.noop import NoOpRerankerConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.llm.chat.echo import EchoChatConfig


class RAGInferenceConfig(AbstractConfig):
    type: Literal["rag_inference"] = "rag_inference"
    query_rewrite_config: Annotated[
        Union[LLMQueryRewriterConfig, NoOpQueryRewriterConfig],
        Field(discriminator="type"),
    ]
    retrieval_config: MultiPathRetrieverConfig
    reranker_config: Annotated[
        Union[LLMRerankerConfig, NoOpRerankerConfig],
        Field(discriminator="type"),
    ]
    llm_config: Annotated[
        Union[OpenAIChatConfig, EchoChatConfig],
        Field(discriminator="type"),
    ]
    
    def build(self):
        return RAGInference(self)
