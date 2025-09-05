from .base import LLMBase
from typing import Union, List, Dict, Any, Optional


class HuggingFaceLLM(LLMBase):
    """
    HuggingFace embedding model implementation
    Pure embedding operations - no business logic
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize base class with embedding support
        super().__init__(model_name, task_types=['embedding'], **kwargs)
        
        self.device = device
        self.cache_folder = cache_folder
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        
        self._client = None
        
        # Initialize embedding model
        self._init_model()
    
    def _init_model(self):
        """Initialize sentence transformer for embedding"""
        try:
            import sentence_transformers
            self._client = sentence_transformers.SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device,
                **self.model_kwargs
            )
            
            self.logger.info(f"HuggingFace LLM initialized: {self.model_name}")
            
        except ImportError:
            raise ImportError("sentence-transformers required for embedding task")
    
    def _embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        文本嵌入生成
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            嵌入向量或嵌入向量列表
        """
        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        try:
            embeddings = self.embed_documents(text_list)
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            self.logger.error(f"Embedding failed: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # Clean texts
            texts = [text.replace("\n", " ") for text in texts]
            
            embeddings = self._client.encode(
                texts,
                convert_to_tensor=False,
                **self.encode_kwargs
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"Document embedding failed: {str(e)}")
            raise RuntimeError(f"Document embedding failed: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "device": self.device,
            "cache_folder": self.cache_folder,
            "provider": "huggingface",
            "model_type": "sentence_transformer"
        })
        return info
    
    # ==================== NOT SUPPORTED METHODS ====================
    
    def _chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support chat"""
        raise NotImplementedError("HuggingFace embedding models do not support chat")
    
    def _stream_chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support streaming chat"""
        raise NotImplementedError("HuggingFace embedding models do not support streaming chat")
    
    def _rerank(self, query: str, documents: List[str], top_k: Optional[int] = None):
        """HuggingFace embedding models don't support reranking"""
        raise NotImplementedError("HuggingFace embedding models do not support reranking")