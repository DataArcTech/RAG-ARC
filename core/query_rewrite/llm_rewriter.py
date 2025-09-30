from typing import Dict, Any, TYPE_CHECKING
from .base import AbstractQueryRewriter
from core.prompts.query_rewrite_prompt import QUERY_REWRITE_USER_PROMPT

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
        # Build LLM from sub-config following framework pattern
        # Accepts any ChatLLMBase implementation (OpenAI, Qwen, HuggingFace, etc.)
        self.chat_llm = config.chat_llm_config.build()

    def rewrite_query(
        self,
        query: str,
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
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Get instruction from config (with default value set in config)
        instruction = self.config.instruction
        logger.info("Using instruction from config")

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": QUERY_REWRITE_USER_PROMPT.format(query=query)}
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