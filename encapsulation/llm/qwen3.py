from .base import LLMBase
from .document import Document
from typing import List, Dict, Any, Optional, Tuple, Union
import logging


class QwenLLM(LLMBase):
    """
    Qwen3 reranker model implementation
    Pure reranking operations - no business logic
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/qwen_reranker_0.6B",
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize base class with reranking support
        super().__init__(model_name, task_types=['rerank'], **kwargs)
        
        self.device = device
        self.cache_folder = cache_folder
        self.model_kwargs = model_kwargs or {}
        
        self._client = None
        self._tokenizer = None
        
        # Initialize reranker model
        self._init_model()
    
    def _init_model(self):
        """Initialize reranker model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_folder,
                trust_remote_code=True,
                padding_side='left'
            )
            
            self._client = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_folder,
                trust_remote_code=True,
                dtype=torch.float16,
                **self.model_kwargs
            )
            self._client.to(self.device)
            
            # Initialize Qwen-specific tokens
            self.token_false_id = self._tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
            
            # Qwen conversation template
            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            # Tokenize prefix and suffix
            self.prefix_tokens = self._tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self._tokenizer.encode(self.suffix, add_special_tokens=False)
            
            # Default instruction
            self.instruction = "Given the user query, retrieve the relevant passages"
            
            self.logger.info(f"Qwen LLM (Reranker) initialized: {self.model_name}")
            
        except ImportError:
            raise ImportError("transformers required for reranking task")
    
    def format_instruction(self, instruction, query, doc):
        """Format instruction with query and document"""
        if instruction is None:
            instruction = self.instruction
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
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
    
    def compute_logits(self, inputs, **kwargs):
        """Compute logits for yes/no tokens"""
        import torch
        with torch.no_grad():
            batch_scores = self._client(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores
    
    def compute_scores(self, pairs, instruction=None, **kwargs):
        """Compute scores for query-document pairs"""
        pairs = [self.format_instruction(instruction, query, doc) for query, doc in pairs]
        inputs = self._process_inputs(pairs)
        scores = self.compute_logits(inputs)
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
                batch_scores = self.compute_scores(batch_pairs, instruction)
                all_scores.extend(batch_scores)
            
            # Create (index, score) pairs and sort by score descending
            ranked_docs = [(i, float(score)) for i, score in enumerate(all_scores)]
            ranked_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Apply top_k filtering if specified
            if top_k is not None:
                ranked_docs = ranked_docs[:top_k]
            
            return ranked_docs
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {str(e)}")
            raise RuntimeError(f"Reranking failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "device": self.device,
            "cache_folder": self.cache_folder,
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
    
    def _embed(self, texts):
        """Qwen reranker models don't support embedding"""
        raise NotImplementedError("Qwen reranker models do not support embedding")