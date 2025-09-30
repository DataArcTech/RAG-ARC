from .base import RerankLLMBase
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from encapsulation.llm.utils.openai_client import create_openai_sync_client, create_openai_async_client
from encapsulation.llm.utils.huggingface_client import create_transformers_client, setup_rerank_tokens
import logging

if TYPE_CHECKING:
    from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)


class QwenRerankLLM(RerankLLMBase):
    """
    Qwen reranker model implementation for advanced chunk relevance scoring.

    This class provides a complete reranking solution using Qwen's causal language model approach,
    supporting sophisticated query-chunk relevance assessment with yes/no token prediction.
    Optimized for high-precision chunk ranking with customizable instruction templates.

    Key features:
    - Causal LM-based reranking with yes/no token scoring
    - Flexible instruction template system for different domains
    - Batch processing with configurable batch sizes
    - Multi-device support with automatic GPU acceleration
    - Advanced tokenization with prefix/suffix template handling
    - Probability-based scoring with log-softmax normalization

    Main parameters:
        config (AbstractConfig): Configuration containing model path, device, instruction settings, etc.
        _client: Lazy-initialized AutoModelForCausalLM instance
        _tokenizer: Lazy-initialized AutoTokenizer instance

    Core methods:
        - rerank/_rerank: Main chunk reranking interface
        - compute_scores: Batch scoring for query-chunk pairs
        - compute_logits: Low-level logit computation with yes/no tokens
        - format_instruction: Template formatting for different query types

    Scoring mechanism:
        - Uses yes/no token prediction for binary relevance assessment
        - Applies conversation template with system/user/assistant structure
        - Computes log-softmax probabilities for robust scoring
        - Supports custom instructions for domain-specific ranking

    Performance considerations:
        - GPU acceleration with automatic memory management
        - Batch processing for improved throughput
        - Template pre-tokenization for reduced overhead
        - Configurable max length for memory efficiency

    Configuration options:
        - model_name: Qwen model identifier (qwen_reranker_0.6B, custom models)
        - device: Target device (cpu, cuda, cuda:0, etc.)
        - cache_folder: Local cache directory for models
        - model_kwargs: Additional model initialization parameters
        - batch_size: Processing batch size for optimal memory usage
        - instruction: Default instruction template for ranking tasks
    """

    def __init__(self, config):
        """Initialize Qwen Rerank with loading method support"""
        super().__init__(config)
        # Cache config values to avoid repeated getattr calls
        self.model_name = getattr(self.config, 'model_name', 'Qwen/qwen_reranker_0.6B')
        self.device = getattr(self.config, 'device', 'cpu')
        self.cache_folder = getattr(self.config, 'cache_folder', None)
        self.instruction = getattr(self.config, 'instruction', "Given the user query, retrieve the relevant passages")
        self.loading_method = getattr(self.config, 'loading_method', 'huggingface')

        # Initialize client based on loading method
        if self.loading_method == 'openai':
            self.client = create_openai_sync_client(self.config)
            self.async_client = create_openai_async_client(self.config)
        elif self.loading_method == 'huggingface':
            # For Qwen rerank, we get (model, tokenizer) tuple
            self.client, self._tokenizer = create_transformers_client(self.config)
            self._setup_qwen_tokens()
        else:
            raise ValueError(f"Unsupported loading method: {self.loading_method}")

    def _setup_qwen_tokens(self):
        """Setup Qwen-specific tokens and templates using generic utility function"""
        rerank_setup = setup_rerank_tokens(self._tokenizer, self.config, model_type="qwen")

        # Set instance attributes from utility function result
        self.token_false_id = rerank_setup['token_false_id']
        self.token_true_id = rerank_setup['token_true_id']
        self.prefix = rerank_setup['prefix']
        self.suffix = rerank_setup['suffix']
        self.prefix_tokens = rerank_setup['prefix_tokens']
        self.suffix_tokens = rerank_setup['suffix_tokens']

    def _format_instruction(self, instruction, query, chunk):
        """Format instruction with query and chunk"""
        if instruction is None:
            instruction = self.instruction
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Chunk>: {chunk}"
        return output

    def _process_inputs(self, pairs):
        """Process input pairs with proper tokenization and padding"""
        # Tokenize pairs without padding first
        out = self._tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=4096 - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        # Add prefix and suffix tokens
        for i, input_ids in enumerate(out['input_ids']):
            out['input_ids'][i] = self.prefix_tokens + input_ids + self.suffix_tokens

        # Apply padding
        out = self._tokenizer.pad(
            out,
            padding=True,
            return_tensors="pt",
            max_length=4096
        )

        # Move to device
        for key in out:
            out[key] = out[key].to(self.device)

        return out

    def _compute_logits(self, inputs, **kwargs):
        """Compute logits for yes/no tokens"""
        import torch

        with torch.no_grad():
            batch_scores = self.client(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores

    def _compute_scores(self, pairs, instruction=None, **kwargs):
        """Compute scores for query-chunk pairs"""
        pairs = [self._format_instruction(instruction, query, chunk) for query, chunk in pairs]
        inputs = self._process_inputs(pairs)
        scores = self._compute_logits(inputs)
        return scores

    def rerank(
        self,
        query: str,
        chunks: List['Chunk'],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Internal reranking implementation using Qwen causal LM approach

        Args:
            query: Query text
            chunks: List of Chunk objects
            top_k: Return top k results

        Returns:
            List of (chunk_index, score) tuples sorted by relevance
        """
        try:
            # Default batch size and instruction for internal method
            batch_size = 8
            instruction = None

            # Create query-chunk pairs using Chunk.content
            pairs = [(query, doc.content) for doc in chunks]

            # Compute scores in batches
            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                batch_scores = self._compute_scores(batch_pairs, instruction)
                all_scores.extend(batch_scores)

            # Create (index, score) pairs and sort by score descending
            ranked_chunks = [(i, float(score)) for i, score in enumerate(all_scores)]
            ranked_chunks.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k filtering if specified
            if top_k is not None:
                ranked_chunks = ranked_chunks[:top_k]

            return ranked_chunks

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise RuntimeError(f"Reranking failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": self.model_name,
            "device": self.device,
            "cache_folder": self.cache_folder,
            "instruction": self.instruction,
            "provider": "huggingface",
            "model_type": "sequence_classification",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown')
        }