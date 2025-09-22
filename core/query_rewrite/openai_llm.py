from typing import Dict, Any, Optional
from .base import AbstractQueryRewriter

import logging

logger = logging.getLogger(__name__)


class OpenAIQueryRewriter(AbstractQueryRewriter):
    """
    OpenAI-based query rewriter for RAG systems.

    Uses OpenAI's language models to rewrite user queries for improved retrieval
    effectiveness. The rewriter can expand ambiguous queries, add context,
    rephrase for better semantic matching, and generate multiple query variations.

    RAG Pipeline Position:
        User Query → Query Rewrite → Retrieval → Rerank → LLM Generate Answer
                     ↑ This component

    Key features:
    - Configurable system prompts for different rewriting strategies
    - Temperature control for creativity vs consistency
    - Token limit management for efficient processing
    - Integration with RAG-ARC LLM infrastructure
    """

    def __init__(self, config):
        super().__init__(config)
        # Build LLM from sub-config following framework pattern
        self.openai_llm = config.openai_llm_config.build()

    def rewrite_query(
        self,
        query: str,
         instruction: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Rewrite a query using OpenAI's language model.

        Primary configuration (instruction, temperature, max_tokens) comes from config.
        Instruction parameter allows runtime override.

        Args:
            query: Original user query to rewrite
            instruction: Optional override for config instruction template
            **kwargs: Additional parameters (can override max_tokens, temperature)

        Returns:
            Rewritten query string optimized for retrieval

        Raises:
            ValueError: If query is empty or invalid
            Exception: If LLM call fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Use provided instruction or default system message
        default_instruction = (
            "You are a query rewriting assistant for a retrieval system. "
            "Your task is to rewrite user queries to improve information retrieval. "
            "Rewrite the query to be more specific, add relevant context, and use "
            "terminology that would appear in documents. Keep the rewritten query "
            "concise and focused. Return only the rewritten query, no explanations."
        )

        if instruction is not None:
            final_instruction = instruction  # instruction override from parameter
            logger.info("Using instruction override from parameter")
        else:
            # Get instruction from config, use default if None/empty
            config_instruction = getattr(self.config, "instruction", None)
            if config_instruction:
                final_instruction = config_instruction
                logger.info("Using instruction from config")
            else:
                final_instruction = default_instruction
                logger.debug("Using default instruction")

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": final_instruction},
            {"role": "user", "content": f"Rewrite this query for better retrieval: {query}"}
        ]

        try:
            # Use LLM to rewrite query - pass all parameters to encapsulation layer
            rewritten = self.openai_llm.chat(
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
            "type": "openai_query_rewriter",
            "llm_info": self.openai_llm.get_model_info(),
            "supports_instruction_override": True,
            "fallback_strategy": "original_query"
        }