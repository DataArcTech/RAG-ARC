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
        "However, if the query refers to a specific entity/product but is under-specified (e.g., abbreviated product name), "
        "and the conversation history contains a more specific/full name, minimally disambiguate by using that fuller name. "
        "For Chinese proper nouns (company/product names), you may include both Simplified and Traditional variants ONLY for those names "
        "when it can improve matching against source documents. "
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
    use_history_for_rewrite: bool = Field(
        default=True,
        description="Whether to include conversation history when rewriting ambiguous follow-up questions.",
    )
    rewrite_history_max_chars: int = Field(
        default=2000,
        ge=0,
        le=20000,
        description="Max characters of history_text passed into the query rewrite prompt (0 disables history inclusion).",
    )

    def build(self) -> LLMQueryRewriter:
        return LLMQueryRewriter(self)
