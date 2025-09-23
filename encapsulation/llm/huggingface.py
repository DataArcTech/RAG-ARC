from encapsulation.llm.base import LLMBase
from typing import Union, List, Dict, Any, Optional, Tuple
import os

from functools import cached_property

from framework.shared_module_decorator import shared_module
from encapsulation.data_model.schema import Document


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
        self._setup_logging()
        # Initialize client immediately since we always need it for embeddings
        self._create_client()

    def _create_client(self):
        """Create HuggingFace SentenceTransformer client"""
        try:
            model_kwargs = self.config.model_kwargs or {}
            
            # 如果使用国内镜像源，设置环境变量
            if self.config.use_china_mirror:
                # 设置HF镜像源
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.logger.info("Set HF_ENDPOINT to https://hf-mirror.com")
            
            # 如果只使用本地文件（离线模式），添加相应参数
            if self.config.local_files_only:
                model_kwargs['local_files_only'] = True
            
            # 注意：SentenceTransformer.__init__ 不接受 mirror 参数
            # 我们通过环境变量 HF_ENDPOINT 来设置镜像源，在这里导入sentence_transformers，一些机器中会因为环境变量设置太晚而无法生效
            import sentence_transformers
            self._client = sentence_transformers.SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_folder,
                device=self.config.device,
                **model_kwargs
            )
            
            self.logger.info(f"HuggingFace LLM initialized: {self.config.model_name}")
            if self.config.use_china_mirror:
                self.logger.info("Using China mirror for model download")
            if self.config.local_files_only:
                self.logger.info("Running in offline mode (local files only)")
            
        except ImportError as e:
            self.logger.error(f"sentence-transformers required for embedding task: {str(e)}")
            raise ImportError("sentence-transformers required for embedding task. Please install it with: pip install sentence-transformers") from e
        except Exception as e:
            self.logger.error(f"Failed to initialize HuggingFace model: {str(e)}")
            raise RuntimeError(f"Failed to initialize HuggingFace model: {str(e)}") from e
    
    def _embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Internal embedding implementation"""
        if isinstance(texts, str):
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # Clean texts
            texts = [text.replace("\n", " ") for text in texts]
            if self._client is not None:
                if self.config.multi_process:
                    import sentence_transformers
                    try:
                        pool = self._client.start_multi_process_pool()
                        embeddings = self._client.encode(texts, pool)
                        sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
                        return embeddings.tolist()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            self.logger.error(f"CUDA Out of Memory in multi-process mode!")
                            self.logger.error(f"Try: config.multi_process = False or reduce batch_size")
                            raise RuntimeError(f"CUDA Out of Memory in multi-process mode. Disable multi_process or reduce batch_size.") from e
                        else:
                            raise

                else:
                    encode_kwargs = self.config.encode_kwargs or {}
                    try:
                        embeddings = self._client.encode(
                            texts,
                            **encode_kwargs
                        )
                        return embeddings.tolist()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            current_batch_size = encode_kwargs.get('batch_size', 'default')
                            self.logger.error(f"CUDA Out of Memory! Current batch_size: {current_batch_size}")
                            self.logger.error(f"Try reducing batch_size in encode_kwargs (e.g., batch_size=8, 4, or 1)")
                            raise RuntimeError(f"CUDA Out of Memory. Please reduce batch_size. Current: {current_batch_size}") from e
                        else:
                            raise
            else:
                raise RuntimeError("Model client not initialized")
        
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
            "device": getattr(self.config, 'device', 'cpu'),
            "cache_folder": getattr(self.config, 'cache_folder', None),
            "provider": "huggingface",
            "model_type": "sentence_transformer",
            "use_china_mirror": self.config.use_china_mirror,
            "local_files_only": self.config.local_files_only
        })
        return info
    
    # ==================== NOT SUPPORTED METHODS ====================
    
    def _chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support chat"""
        raise NotImplementedError("HuggingFace embedding models do not support chat")
    
    def _stream_chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support streaming chat"""
        raise NotImplementedError("HuggingFace embedding models do not support streaming chat")

    async def _achat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> str:
        """HuggingFace embedding models don't support async chat"""
        raise NotImplementedError("HuggingFace embedding models do not support async chat")

    async def _astream_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs):
        """HuggingFace embedding models don't support async streaming chat"""
        raise NotImplementedError("HuggingFace embedding models do not support async streaming chat")
        yield  # This will never be reached, but satisfies the async generator type

    async def _aembed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Async embedding - just calls sync version for now"""
        return self._embed(texts)

    def _rerank(self, query: str, documents: List['Document'], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """HuggingFace embedding models don't support reranking"""
        raise NotImplementedError("HuggingFace embedding models do not support reranking")