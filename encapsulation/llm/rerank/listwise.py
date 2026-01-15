from .base import RerankLLMBase
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import logging
import json
import re

from core.utils.json_extract import extract_last_json_array_from_text
from core.prompts import LISTWISE_RERANK_DEFAULT_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from encapsulation.data_model.schema import Chunk
    from encapsulation.llm.chat.base import ChatLLMBase

logger = logging.getLogger(__name__)


class ListwiseRerankLLM(RerankLLMBase):
    """
    Listwise reranker model implementation using LLM for ranking.

    This class provides a complete listwise reranking solution using chat LLMs,
    where the model ranks all documents at once by outputting a ranked list of indices.
    Optimized for high-quality ranking with reasoning capabilities.

    Key features:
    - Listwise ranking approach (ranks all documents together)
    - Reasoning-based ranking with step-by-step thinking
    - Flexible prompt template system for different domains
    - Batch processing support for large document sets
    - JSON-based output parsing for robust index extraction
    - Fallback to original order on parsing errors

    Main parameters:
        config (AbstractConfig): Configuration containing chat LLM config, prompt template, etc.
        chat_llm: ChatLLMBase instance for LLM interactions

    Core methods:
        - rerank: Main chunk reranking interface
        - _format_prompt: Template formatting for ranking tasks
        - _parse_ranking_output: Extract ranked indices from LLM output
        - _call_llm: Call chat LLM for ranking

    Ranking mechanism:
        - Formats query and documents into a structured prompt
        - Asks LLM to reason about relevance and output ranked indices
        - Parses JSON output to extract ranking order
        - Assigns scores based on ranking position (higher rank = higher score)

    Performance considerations:
        - Uses chat LLM for better responsiveness
        - Configurable prompt template for different domains
        - Error handling with fallback to original order
        - Supports both sync and async operations

    Configuration options:
        - chat_llm_config: Configuration for the chat LLM
        - prompt_template: Custom prompt template (optional)
    """

    def __init__(self, config):
        """Initialize Listwise Rerank with chat LLM"""
        super().__init__(config)
        # Build chat LLM from config
        self.chat_llm = config.chat_llm_config.build()
        self.default_prompt_template = getattr(self.config, 'prompt_template', None)

    def _format_prompt(self, query: str, chunks: List['Chunk'], top_k: int, prompt_template: Optional[str] = None) -> str:
        """
        Format the ranking prompt with query and documents.

        Args:
            query: User query
            chunks: List of Chunk objects
            top_k: Number of top documents to return
            prompt_template: Custom prompt template (optional)

        Returns:
            Formatted prompt string
        """
        # Format documents with indices
        newline_pattern = r'\n+'
        doc_str = ''.join([
            f"[{i + 1}]. {re.sub(newline_pattern, ' ', chunk.content)}\n\n"
            for i, chunk in enumerate(chunks)
        ])

        # Use custom template or default
        if prompt_template is None:
            prompt_template = self.default_prompt_template

        if prompt_template is None:
            # Default prompt template
            prompt_template = LISTWISE_RERANK_DEFAULT_PROMPT_TEMPLATE

        # Format the prompt
        prompt = prompt_template.format(QUERY=query, DOC_STR=doc_str, TOPK=top_k)
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the chat LLM for ranking.

        Args:
            prompt: Formatted prompt string

        Returns:
            Complete response string
        """
        try:
            # Use chat LLM's chat method
            messages = [{"role": "user", "content": prompt}]
            result = self.chat_llm.chat(messages=messages)
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return "error"

    def _parse_ranking_output(self, output_str: str, num_chunks: int, top_k: int) -> List[int]:
        """
        Parse the LLM output to extract ranked indices.

        Args:
            output_str: Raw LLM output string
            num_chunks: Total number of chunks
            top_k: Number of top documents to return

        Returns:
            List of ranked indices (1-based from LLM, converted to 0-based)
        """
        if output_str is None or output_str == "error":
            logger.warning("Invalid output, using default ranking")
            return list(range(min(top_k, num_chunks)))

        try:
            # Remove thinking tags if present
            if "</think>" in output_str:
                output_str = output_str.split("</think>")[-1]

            idxs_str = extract_last_json_array_from_text(output_str)
            if idxs_str:
                idxs = json.loads(idxs_str)
                
                # Validate indices
                if isinstance(idxs, list) and len(idxs) > 0:
                    # Convert 1-based to 0-based and validate range
                    valid_idxs = []
                    for idx in idxs:
                        if isinstance(idx, int) and 1 <= idx <= num_chunks:
                            valid_idxs.append(idx - 1)  # Convert to 0-based
                    
                    if valid_idxs:
                        return valid_idxs[:top_k]
            
            logger.warning(f"Failed to parse ranking output: {output_str[:200]}")
            return list(range(min(top_k, num_chunks)))

        except Exception as e:
            logger.error(f"Error parsing ranking output: {str(e)}")
            return list(range(min(top_k, num_chunks)))

    def rerank(
        self,
        query: str,
        chunks: List['Chunk'],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Internal reranking implementation using listwise ranking approach.

        Args:
            query: Query text
            chunks: List of Chunk objects
            top_k: Return top k results (defaults to all chunks)

        Returns:
            List of (chunk_index, score) tuples sorted by relevance
        """
        try:
            if not chunks:
                return []

            # Default top_k to number of chunks
            if top_k is None:
                top_k = len(chunks)
            
            top_k = min(top_k, len(chunks))

            # Format prompt
            prompt = self._format_prompt(query, chunks, top_k)

            # Call LLM
            output_str = self._call_llm(prompt)

            # Parse ranking output
            ranked_indices = self._parse_ranking_output(output_str, len(chunks), top_k)

            # Guarantee `top_k` unique indices for downstream coverage.
            # Some LLMs may return fewer than TOPK indices even when asked; fill the remainder
            # with the original order (excluding already chosen) to keep behavior stable and debuggable.
            if len(ranked_indices) < top_k:
                seen: set[int] = set()
                deduped: list[int] = []
                for idx in ranked_indices:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    deduped.append(idx)
                for idx in range(len(chunks)):
                    if len(deduped) >= top_k:
                        break
                    if idx in seen:
                        continue
                    seen.add(idx)
                    deduped.append(idx)
                if len(deduped) != len(ranked_indices):
                    logger.info(
                        "Listwise rerank returned %d/%d indices from LLM; filled to %d with fallback order",
                        len(ranked_indices),
                        top_k,
                        len(deduped),
                    )
                ranked_indices = deduped[:top_k]

            # Assign scores based on ranking position (higher rank = higher score)
            # Score = top_k - position (so first ranked gets highest score)
            ranked_chunks = [
                (idx, float(top_k - i))
                for i, idx in enumerate(ranked_indices)
            ]

            return ranked_chunks

        except Exception as e:
            logger.error(f"Listwise reranking failed: {str(e)}")
            # Fallback: return original order with decreasing scores
            return [(i, float(len(chunks) - i)) for i in range(min(top_k or len(chunks), len(chunks)))]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        chat_llm_info = self.chat_llm.get_model_info()

        return {
            "chat_llm_info": chat_llm_info,
            "model_type": "listwise_rerank",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown'),
            "has_custom_prompt": self.default_prompt_template is not None
        }
