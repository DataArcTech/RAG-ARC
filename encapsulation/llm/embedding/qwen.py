import logging
from typing import Union, List, Dict, Any, TYPE_CHECKING

from encapsulation.llm.embedding.base import EmbeddingLLMBase
from encapsulation.llm.utils.openai_client import create_openai_sync_client, create_openai_async_client
from encapsulation.llm.utils.huggingface_client import create_sentence_transformer_client
from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig

logger = logging.getLogger(__name__)


@shared_module
class QwenEmbeddingLLM(EmbeddingLLMBase):
    """
    Qwen embedding model implementation for high-performance text vectorization.

    This class provides a complete embedding solution using HuggingFace's SentenceTransformers library,
    supporting various pre-trained Qwen models with flexible device configuration and caching capabilities.
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
        - encode_kwargs: Encoding-specific parameters (batch_size, show_progress_bar, etc.)
    """


    def __init__(self, config: "QwenEmbeddingConfig"):
        """Initialize Qwen Embedding with loading method support"""
        super().__init__(config)
        self.loading_method = getattr(self.config, 'loading_method', 'huggingface')

        # Initialize client based on loading method
        if self.loading_method == 'openai':
            self.client = create_openai_sync_client(self.config)
            self.async_client = create_openai_async_client(self.config)
        elif self.loading_method == 'huggingface':
            self.client = create_sentence_transformer_client(self.config)
            self.async_client = None  # HuggingFace uses asyncio.to_thread wrapper
        else:
            raise ValueError(f"Unsupported loading method: {self.loading_method}")

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate text embeddings"""
        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        try:
            embeddings = self.embed_chunks(text_list)
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            logger.error(f"Text embedding failed: {str(e)}")
            raise

    async def aembed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate text embeddings asynchronously"""
        import asyncio

        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        try:
            # Run the synchronous embedding in a thread pool
            embeddings = await asyncio.to_thread(self.embed_chunks, text_list)
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            logger.error(f"Async text embedding failed: {str(e)}")
            raise

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple chunks"""
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
            logger.error(f"Chunk embedding failed: {str(e)}")
            raise RuntimeError(f"Chunk embedding failed: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_chunks([text])[0]

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