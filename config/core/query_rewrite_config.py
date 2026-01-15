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
        "Your job is to improve retrieval while preserving user intent. "
        "If the user intent is already clear and the query is retrieval-ready, return the original query verbatim. "
        "Do NOT translate or switch languages. "
        "Do NOT expand the question into additional subquestions or add generic terms that may drift intent. "
        "You may only make minimal edits (e.g., normalize wording, add up to a few synonyms) when it clearly helps retrieval. "
        "Return only the final query text with no explanations."
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
