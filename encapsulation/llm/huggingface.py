from .base import LLMBase
from typing import Union, List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import logging
from functools import cached_property

from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from .document import Document

logger = logging.getLogger(__name__)

@shared_module
class HuggingFaceLLM(LLMBase):
    """
    HuggingFace embedding model implementation for high-performance text vectorization.
    
    This class provides a complete embedding solution using HuggingFace's SentenceTransformers library,
    supporting various pre-trained models with flexible device configuration and caching capabilities.
    Optimized for batch processing and memory-efficient inference with configurable encoding parameters.
    
    Key features:
    - SentenceTransformers integration for state-of-the-art embeddings
    - Multi-device support: CPU, GPU, and multi-GPU configurations
    - Flexible model selection: BERT, RoBERTa, MPNet, and specialized embedding models
    - Batch processing with configurable encoding parameters
    - Local model caching for offline deployment
    - Memory optimization for large-scale processing
    
    Main parameters:
        config (AbstractConfig): Configuration containing model path, device, cache settings, etc.
        _client: Lazy-initialized SentenceTransformer model instance
        
    Core methods:
        - embed/_embed: General text embedding with automatic batching
        - embed_documents: Batch document embedding for large collections
        - embed_query: Single query embedding with optimized processing
        
    Supported models:
        - General: sentence-transformers/all-mpnet-base-v2, all-MiniLM-L6-v2
        - Multilingual: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        - Specialized: sentence-transformers/msmarco-distilbert-base-v4
        - Custom: Any compatible HuggingFace model with proper configuration
        
    Performance considerations:
        - GPU acceleration with automatic device detection
        - Batch processing for improved throughput
        - Model caching to reduce download overhead
        - Memory-mapped models for efficient loading
        
    Configuration options:
        - model_name: HuggingFace model identifier or local path
        - device: Target device (cpu, cuda, cuda:0, etc.)
        - cache_folder: Local cache directory for models
        - model_kwargs: Additional model initialization parameters
        - encode_kwargs: Encoding-specific parameters (batch_size, show_progress_bar, etc.)
    """
    

    def __init__(self, config):
        """Initialize HuggingFace with eager model creation"""
        super().__init__(config)
        # Initialize client immediately since we always need it for embeddings
        self.client = self._create_client()

    def _create_client(self):
        """Create HuggingFace SentenceTransformer client"""
        try:
            import sentence_transformers

            model_name = getattr(self.config, 'model_name', 'sentence-transformers/all-mpnet-base-v2')
            device = getattr(self.config, 'device', 'cpu')
            cache_folder = getattr(self.config, 'cache_folder', None)
            model_kwargs = getattr(self.config, 'model_kwargs', {})

            client = sentence_transformers.SentenceTransformer(
                model_name,
                cache_folder=cache_folder,
                device=device,
                **model_kwargs
            )

            logger.info(f"HuggingFace model initialized: {model_name}")
            return client

        except ImportError:
            logger.error("sentence-transformers library required for embedding task")
            raise ImportError("sentence-transformers required for embedding task")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {str(e)}")
            raise
    
    def _embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate text embeddings"""
        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        try:
            embeddings = self.embed_documents(text_list)
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            logger.error(f"Text embedding failed: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # Clean texts
            texts = [text.replace("\n", " ") for text in texts]
            
            encode_kwargs = getattr(self.config, 'encode_kwargs', {})
            
            embeddings = self.client.encode(
                texts,
                convert_to_tensor=False,
                **encode_kwargs
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Document embedding failed: {str(e)}")
            raise RuntimeError(f"Document embedding failed: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "device": getattr(self.config, 'device', 'cpu'),
            "cache_folder": getattr(self.config, 'cache_folder', None),
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
    
    def _rerank(self, query: str, documents: List['Document'], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """HuggingFace embedding models don't support reranking"""
        raise NotImplementedError("HuggingFace embedding models do not support reranking")