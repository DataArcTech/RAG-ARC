from .base import EmbeddingLLMBase
from typing import Union, List, Dict, Any, Optional
import openai
import logging

logger = logging.getLogger(__name__)


class OpenAIEmbeddingLLM(EmbeddingLLMBase):
    """
    OpenAI Embedding LLM implementation for high-quality text vectorization.

    This class provides a complete embedding solution using OpenAI's API,
    supporting various embedding models with configurable dimensions and parameters.
    Optimized for production use with automatic retry logic and batch processing.

    Key features:
    - Multiple embedding model support: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
    - Configurable embedding dimensions for supported models
    - Batch processing for improved efficiency
    - Automatic retry logic and timeout handling
    - Flexible API endpoint configuration (OpenAI or compatible)

    Configuration options:
        - api_key: OpenAI API key or compatible service key
        - base_url: API endpoint (defaults to OpenAI, supports custom servers)
        - model_name: Target embedding model identifier
        - embedding_dimensions: Custom embedding size (if supported)
        - organization: OpenAI organization ID
        - timeout: Request timeout in seconds
        - max_retries: Automatic retry attempts
    """

    def __init__(self, config):
        """Initialize OpenAI Embedding with eager client creation"""
        super().__init__(config)
        # Cache config values to avoid repeated getattr calls
        self.model_name = getattr(self.config, 'model_name', 'text-embedding-ada-002')
        self.embedding_dimensions = getattr(self.config, 'embedding_dimensions', None)
        # Initialize client immediately since we always need it for API calls
        self.client = self._create_client()
        self.async_client = self._create_async_client()

    def _create_client(self):
        """Create OpenAI client"""
        # Extract OpenAI-specific config parameters
        api_key = getattr(self.config, 'api_key', None)
        base_url = getattr(self.config, 'base_url', None)
        organization = getattr(self.config, 'organization', None)
        max_retries = getattr(self.config, 'max_retries', 3)
        timeout = getattr(self.config, 'timeout', 60.0)

        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                max_retries=max_retries,
                timeout=timeout
            )
            logger.info(f"OpenAI embedding client initialized: {self.model_name}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def _create_async_client(self):
        """Create OpenAI async client"""
        try:
            # Extract OpenAI-specific config parameters (same as sync client)
            api_key = getattr(self.config, 'api_key', None)
            base_url = getattr(self.config, 'base_url', None)
            organization = getattr(self.config, 'organization', None)
            max_retries = getattr(self.config, 'max_retries', 3)
            timeout = getattr(self.config, 'timeout', 30)

            async_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                max_retries=max_retries,
                timeout=timeout
            )
            logger.info(f"OpenAI async embedding client initialized: {self.model_name}")
            return async_client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI async client: {str(e)}")
            raise

    # ==================== EMBEDDING IMPLEMENTATION ====================

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using OpenAI
        """
        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        # Validate inputs
        for text in text_list:
            if not self._validate_input(text):
                raise ValueError(f"Text validation failed: {text}")

        try:
            # Clean texts - remove newlines
            cleaned_texts = [text.replace("\n", " ") for text in text_list]

            # Create embedding request
            embedding_kwargs = {}
            if self.embedding_dimensions:
                embedding_kwargs['dimensions'] = self.embedding_dimensions

            response = self.client.embeddings.create(
                model=self.model_name,
                input=cleaned_texts,
                **embedding_kwargs
            )

            # Extract embeddings - handle different response formats
            if hasattr(response, 'data') and response.data:
                embeddings = [item.embedding for item in response.data]
            elif isinstance(response, dict) and 'data' in response:
                embeddings = [item['embedding'] for item in response['data']]
            else:
                raise RuntimeError(f"Unexpected response format: {type(response)}")

            # Return single embedding or list based on input
            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise RuntimeError(f"Embedding failed: {str(e)}")

    async def aembed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Async function of generate embeddings
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        for text in text_list:
            if not self._validate_input(text):
                raise ValueError(f"Invalid text: {text}")

        try:
            # Create embedding request
            embedding_kwargs = {}
            if self.embedding_dimensions:
                embedding_kwargs['dimensions'] = self.embedding_dimensions

            embeddings = []
            batch_size = 100
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i:i+batch_size]
                response = await self.async_client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    **embedding_kwargs
                )
                embeddings.extend(data.embedding for data in response.data)

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.error(f"Async embedding failed: {str(e)}")
            raise

    # ==================== CONVENIENCE METHODS ====================

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents - always returns list of embeddings"""
        return self.embed(texts) if isinstance(texts, list) else [self.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        """Embed single query - always returns single embedding"""
        result = self.embed(text)
        return result if isinstance(result, list) and isinstance(result[0], (int, float)) else result[0]

    # ==================== UTILITY METHODS ====================

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        """
        # Safely get client info without forcing initialization
        client = getattr(self, 'client', None)

        return {
            "model": self.model_name,
            "api_base": getattr(client, 'base_url', None) if client else None,
            "organization": getattr(client, 'organization', None) if client else None,
            "max_retries": getattr(client, 'max_retries', None) if client else None,
            "timeout": getattr(client, 'timeout', None) if client else None,
            "embedding_dimensions": self.embedding_dimensions,
            "provider": "openai",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown')
        }

    def _validate_input(self, input_text: str, max_length: Optional[int] = None) -> bool:
        """Validate input text"""
        if not isinstance(input_text, str):
            logger.error("Input must be string type")
            return False

        if not input_text.strip():
            logger.error("Input text cannot be empty")
            return False

        if max_length and len(input_text) > max_length:
            logger.error(f"Input text length exceeds limit: {len(input_text)} > {max_length}")
            return False

        return True