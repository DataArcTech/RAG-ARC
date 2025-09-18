from .base import LLMBase
from .document import Document
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class QwenLLM(LLMBase):
    """
    Qwen3 reranker model implementation for advanced document relevance scoring.
    
    This class provides a complete reranking solution using Qwen's causal language model approach,
    supporting sophisticated query-document relevance assessment with yes/no token prediction.
    Optimized for high-precision document ranking with customizable instruction templates.
    
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
        - rerank/_rerank: Main document reranking interface
        - compute_scores: Batch scoring for query-document pairs
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
        """Initialize Qwen with eager model and tokenizer creation"""
        super().__init__(config)
        # Initialize client immediately since we always need it for reranking
        self.client = self._create_client()

    def _create_client(self):
        """Create Qwen client and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_name = getattr(self.config, 'model_name', 'Qwen/qwen_reranker_0.6B')
            device = getattr(self.config, 'device', 'cpu')
            cache_folder = getattr(self.config, 'cache_folder', None)
            model_kwargs = getattr(self.config, 'model_kwargs', {})

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_folder,
                trust_remote_code=True,
                padding_side='left'
            )

            client = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_folder,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                **model_kwargs
            )
            client.to(device)

            # Initialize Qwen-specific tokens
            self.token_false_id = self._tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

            # Qwen conversation template - get from config or use defaults
            self.prefix = getattr(self.config, 'prefix', "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
            self.suffix = getattr(self.config, 'suffix', "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

            # Tokenize prefix and suffix
            self.prefix_tokens = self._tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self._tokenizer.encode(self.suffix, add_special_tokens=False)

            # Default instruction from config or fallback
            self.instruction = getattr(self.config, 'instruction', "Given the user query, retrieve the relevant passages")

            logger.info(f"Qwen reranker initialized: {model_name}")
            return client

        except ImportError:
            logger.error("transformers library required for reranking task")
            raise ImportError("transformers required for reranking task")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {str(e)}")
            raise
    
    def _format_instruction(self, instruction, query, doc):
        """Format instruction with query and document"""
        if instruction is None:
            instruction = getattr(self.config, 'instruction', "Given the user query, retrieve the relevant passages")
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output
    
    def _process_inputs(self, pairs):
        """Process input pairs with proper tokenization and padding"""
        device = getattr(self.config, 'device', 'cpu')
        
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
            out[key] = out[key].to(device)
        
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
        """Compute scores for query-document pairs"""
        pairs = [self._format_instruction(instruction, query, doc) for query, doc in pairs]
        inputs = self._process_inputs(pairs)
        scores = self._compute_logits(inputs)
        return scores

    def _rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Internal reranking implementation using Qwen causal LM approach
        
        Args:
            query: Query text
            documents: List of Document objects
            top_k: Return top k results
            
        Returns:
            List of (doc_index, score) tuples sorted by relevance
        """
        try:
            # Default batch size and instruction for internal method
            batch_size = 8
            instruction = None
            
            # Create query-document pairs using Document.content
            pairs = [(query, doc.content) for doc in documents]
            
            # Compute scores in batches
            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                batch_scores = self._compute_scores(batch_pairs, instruction)
                all_scores.extend(batch_scores)
            
            # Create (index, score) pairs and sort by score descending
            ranked_docs = [(i, float(score)) for i, score in enumerate(all_scores)]
            ranked_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Apply top_k filtering if specified
            if top_k is not None:
                ranked_docs = ranked_docs[:top_k]
            
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise RuntimeError(f"Reranking failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "device": getattr(self.config, 'device', 'cpu'),
            "cache_folder": getattr(self.config, 'cache_folder', None),
            "provider": "huggingface",
            "model_type": "sequence_classification"
        })
        return info
    
    # ==================== NOT SUPPORTED METHODS ====================
    
    def _chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """Qwen reranker models don't support chat"""
        raise NotImplementedError("Qwen reranker models do not support chat")
    
    def _stream_chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """Qwen reranker models don't support streaming chat"""
        raise NotImplementedError("Qwen reranker models do not support streaming chat")

    async def _achat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """Qwen reranker models don't support async chat"""
        raise NotImplementedError("Qwen reranker models do not support async chat")

    async def _astream_chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """Qwen reranker models don't support async streaming chat"""
        raise NotImplementedError("Qwen reranker models do not support async streaming chat")

    def _embed(self, texts):
        """Qwen reranker models don't support embedding"""
        raise NotImplementedError("Qwen reranker models do not support embedding")

    async def _aembed(self, texts):
        """Qwen reranker models don't support async embedding"""
        raise NotImplementedError("Qwen reranker models do not support async embedding")