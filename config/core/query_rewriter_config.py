from framework.config import AbstractConfig
from core.query_rewrite.openai_llm import OpenAIQueryRewriter
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from typing import Literal

class OpenAIQueryRewriterConfig(AbstractConfig):
    """Configuration for OpenAI Query Rewriter"""
    type: Literal["openai_query_rewriter"] = "openai_query_rewriter"
    openai_llm_config: OpenAIChatConfig
    instruction: str = None

    def build(self) -> OpenAIQueryRewriter:
        return OpenAIQueryRewriter(self)