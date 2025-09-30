from framework.config import AbstractConfig
from core.query_rewrite.llm_rewriter import LLMQueryRewriter
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from typing import Literal, Union

class LLMQueryRewriterConfig(AbstractConfig):
    """
    Configuration for LLM-based Query Rewriter.

    This config accepts any ChatLLMBase implementation through dependency injection,
    making it flexible to use with different LLM providers (OpenAI, Qwen, HuggingFace, etc.).
    """
    type: Literal["llm_query_rewriter"] = "llm_query_rewriter"
    chat_llm_config: OpenAIChatConfig  # Accept any ChatLLM config
    instruction: str = (
        "You are a query rewriting assistant for a retrieval system. "
        "Your task is to rewrite user queries to improve information retrieval. "
        "Rewrite the query to be more specific, add relevant context, and use "
        "terminology that would appear in documents. Keep the rewritten query "
        "concise and focused. Return only the rewritten query, no explanations."
    )

    def build(self) -> LLMQueryRewriter:
        return LLMQueryRewriter(self)