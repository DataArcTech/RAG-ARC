from typing import Dict, Any, TYPE_CHECKING
import re
from .base import AbstractQueryRewriter
from core.prompts.query_rewrite_prompt import (
    QUERY_REWRITE_USER_PROMPT,
    QUERY_REWRITE_USER_PROMPT_WITH_HISTORY,
)
from config.benchmark_mode import benchmark_mode_enabled

import logging

if TYPE_CHECKING:
    from config.core.query_rewrite_config import LLMQueryRewriterConfig

logger = logging.getLogger(__name__)


class LLMQueryRewriter(AbstractQueryRewriter):
    """
    LLM-based query rewriter for RAG systems.

    Uses any chat LLM to rewrite user queries for improved retrieval effectiveness.
    The rewriter can expand ambiguous queries, add context, rephrase for better
    semantic matching, and generate multiple query variations.

    This implementation is LLM-agnostic and works with any ChatLLMBase implementation
    from the encapsulation layer (OpenAI, Qwen, HuggingFace, etc.).

    RAG Pipeline Position:
        User Query → Query Rewrite → Retrieval → Rerank → LLM Generate Answer
                     ↑ This component

    Key features:
    - LLM-agnostic: Works with any ChatLLMBase implementation
    - Configurable system prompts for different rewriting strategies
    - Temperature control for creativity vs consistency
    - Token limit management for efficient processing
    - Integration with RAG-ARC LLM infrastructure via dependency injection
    """

    def __init__(self, config):
        super().__init__(config)
        # In benchmark/experiment mode, query rewrite is disabled; avoid building any LLM clients.
        if benchmark_mode_enabled():
            self.chat_llm = None
            return
        # Build LLM from sub-config following framework pattern
        # Accepts any ChatLLMBase implementation (OpenAI, Qwen, HuggingFace, etc.)
        self.chat_llm = config.chat_llm_config.build()

    def rewrite_query(
        self,
        query: str,
        *,
        history_text: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        Rewrite a query using any chat LLM.

        Primary configuration (instruction, temperature, max_tokens) comes from config.

        Args:
            query: Original user query to rewrite
            **kwargs: Additional parameters (can override max_tokens, temperature)

        Returns:
            Rewritten query string optimized for retrieval

        Raises:
            ValueError: If query is empty or invalid
            Exception: If LLM call fails
        """
        if benchmark_mode_enabled():
            return str(query or "")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = str(query)
        if getattr(self.config, "skip_rewrite_if_contains", None):
            for token in list(getattr(self.config, "skip_rewrite_if_contains") or []):
                if token and token in query:
                    logger.info("Skipping query rewrite because query contains %r", token)
                    return query
        if getattr(self.config, "skip_rewrite_regexes", None):
            for pattern in list(getattr(self.config, "skip_rewrite_regexes") or []):
                if not pattern:
                    continue
                try:
                    if re.search(pattern, query):
                        logger.info("Skipping query rewrite because query matches pattern %r", pattern)
                        return query
                except re.error:
                    logger.warning("Invalid skip rewrite regex ignored: %r", pattern)

        # Get instruction from config (with default value set in config)
        instruction = self.config.instruction
        logger.info("Using instruction from config")

        use_history = bool(getattr(self.config, "use_history_for_rewrite", False))
        max_history_chars = int(getattr(self.config, "rewrite_history_max_chars", 0) or 0)
        history_snippet = None
        if use_history and history_text and max_history_chars > 0:
            # Keep only the tail to bias toward recent turns (history is usually chronological).
            history_snippet = str(history_text)[-max_history_chars:]

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": (
                    QUERY_REWRITE_USER_PROMPT_WITH_HISTORY.format(query=query, history=history_snippet)
                    if history_snippet
                    else QUERY_REWRITE_USER_PROMPT.format(query=query)
                ),
            },
        ]

        try:
            # Use LLM to rewrite query - pass all parameters to encapsulation layer
            rewritten = self.chat_llm.chat(
                messages=messages,
                **kwargs  # Pass through all parameters to encapsulation layer
            )

            # Clean up response (remove quotes, extra whitespace)
            rewritten = rewritten.strip().strip('"').strip("'")

            # Fallback to original if rewrite is empty
            if not rewritten:
                logger.warning("LLM returned empty rewrite, using original query")
                return query

            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            # Return original query as fallback
            logger.warning("Using original query as fallback")
            return query

    def get_rewriter_info(self) -> Dict[str, Any]:
        """
        Get information about this query rewriter's configuration.

        Returns:
            Dictionary containing rewriter information
        """
        return {
            "type": "llm_query_rewriter",
            "llm_info": self.chat_llm.get_model_info(),
            "instruction": self.config.instruction,
            "fallback_strategy": "original_query"
        }
