from .base import EmbeddingLLMBase
from typing import Union, List, Dict, Any
import os
import logging

from framework.shared_module_decorator import shared_module

logger = logging.getLogger(__name__)


@shared_module
class HuggingFaceEmbeddingLLM(EmbeddingLLMBase):
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
        - encode_kwargs: Encoding-specific parameters (batch_size, show_progress_bar, local_files_only, etc.)
    """


    def __init__(self, config):
        """Initialize HuggingFace with eager model creation"""
        super().__init__(config)
        # Initialize client immediately since we always need it for embeddings
        self.client = self._create_client()

    def _create_client(self):
        """Create HuggingFace SentenceTransformer client"""
        try:
            if self.config.use_china_mirror:
               os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
               logger.info("Set HF_ENDPOINT to https://hf-mirror.com")

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

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate text embeddings"""
        # Handle single text vs list
        # use embed_query and embed_documents
        try:
            if isinstance(texts, str):
                return self.embed_query(texts)
            else:
                return self.embed_documents(texts)
        except Exception as e:
            logger.error(f"Text embedding failed: {str(e)}")
            raise

        # is_single = isinstance(texts, str)
        # text_list = [texts] if is_single else texts

        # try:
        #     embeddings = self.embed_documents(text_list)
        #     return embeddings[0] if is_single else embeddings
        # except Exception as e:
        #     logger.error(f"Text embedding failed: {str(e)}")
        #     raise

    async def aembed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate text embeddings asynchronously"""
        import asyncio

        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        try:
            # Run the synchronous embedding in a thread pool
            embeddings = await asyncio.to_thread(self.embed_documents, text_list)
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            logger.error(f"Async text embedding failed: {str(e)}")
            raise

    # def embed_documents(self, texts: List[str]) -> List[List[float]]:
    #     """Embed multiple documents"""
    #     try:
    #         # Clean texts
    #         texts = [text.replace("\n", " ") for text in texts]

    #         encode_kwargs = getattr(self.config, 'encode_kwargs', {})

    #         embeddings = self.client.encode(
    #             texts,
    #             convert_to_tensor=False,
    #             **encode_kwargs
    #         )

    #         return embeddings.tolist()

    #     except Exception as e:
    #         logger.error(f"Document embedding failed: {str(e)}")
    #         raise RuntimeError(f"Document embedding failed: {str(e)}")


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents more concisely."""
        if self.client is None:
            raise RuntimeError("Model client not initialized")

        # 预处理文本
        texts = [text.replace("\n", " ") for text in texts]
        pool = None  # 初始化 pool

        import sentence_transformers
        try:
            if self.config.multi_process:
                pool = self.client.start_multi_process_pool()
                embeddings = self.client.encode(texts, pool)
            else:
                encode_kwargs = self.config.encode_kwargs or {}
                embeddings = self.client.encode(texts, **encode_kwargs)
            
            return embeddings.tolist()

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise 

            if self.config.multi_process:
                error_msg = "CUDA Out of Memory in multi-process mode. Disable multi_process or reduce batch_size."
                logger.error("CUDA Out of Memory in multi-process mode!")
                logger.error("Try: config.multi_process = False")
            else:
                batch_size = (self.config.encode_kwargs or {}).get('batch_size', 'default')
                error_msg = f"CUDA Out of Memory. Please reduce batch_size. Current: {batch_size}"
                logger.error(f"CUDA Out of Memory! Current batch_size: {batch_size}")
                logger.error("Try reducing batch_size in encode_kwargs (e.g., batch_size=8, 4, or 1)")
            
            raise RuntimeError(error_msg) from e

        finally:
            if pool:
                sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)

    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": getattr(self.config, 'model_name', 'unknown'),
            "device": getattr(self.config, 'device', 'cpu'),
            "cache_folder": getattr(self.config, 'cache_folder', None),
            "provider": "huggingface",
            "model_type": "sentence_transformer",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown')
        }