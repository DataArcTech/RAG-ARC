from framework.config import AbstractConfig
from core.query_rewrite.llm_rewriter import LLMQueryRewriter
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from typing import Literal, Union
from pydantic import Field

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
        "Preserve the original query language; do NOT translate or switch languages. "
        "Rewrite the query to be more specific, add relevant context, and use "
        "terminology that would appear in documents. Keep the rewritten query "
        "concise and focused. Return only the rewritten query, no explanations."
    )
    skip_rewrite_if_contains: list[str] = Field(
        default_factory=lambda: ["="],
        description="Skip rewrite when the query contains any of these substrings (e.g., key=value lookups).",
    )
    skip_rewrite_regexes: list[str] = Field(
        default_factory=lambda: [
            r"\\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\\b",
        ],
        description="Skip rewrite when the query matches any of these regex patterns.",
    )

    def build(self) -> LLMQueryRewriter:
        return LLMQueryRewriter(self)
