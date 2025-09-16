from .base import LLMBase
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
import openai
import logging
from functools import cached_property

if TYPE_CHECKING:
    from .document import Document

logger = logging.getLogger(__name__)


class OpenAILLM(LLMBase):
    """
    Unified OpenAI LLM implementation supporting both chat and embeddings capabilities.
    
    This class provides a complete language model solution using OpenAI's API,
    supporting multiple model types and task configurations with intelligent client management.
    Optimized for both conversational AI and text embedding generation with streaming support.
    
    Key features:
    - Dual functionality: Chat completions and text embeddings
    - Multiple model support: GPT-4, GPT-3.5, text-embedding models
    - Streaming chat with token usage tracking
    - Configurable parameters: temperature, max_tokens, dimensions
    - Automatic retry logic and timeout handling
    - Flexible API endpoint configuration (OpenAI or compatible)
    
    Main parameters:
        config (AbstractConfig): Configuration containing API credentials, model settings, etc.
        _client: Lazy-initialized OpenAI client instance
        
    Core methods:
        - chat/_chat: Standard chat completion
        - stream_chat/_stream_chat: Streaming chat with real-time responses  
        - embed/_embed: Text embedding generation
        - embed_documents/embed_query: Convenience methods for different use cases
        
    Supported models:
        - Chat: gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini
        - Embeddings: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
        - Custom endpoints: Compatible with VLLM, LocalAI, and other OpenAI-compatible APIs
        
    Performance considerations:
        - Lazy client initialization for faster startup
        - Connection pooling and automatic retries
        - Streaming support for real-time applications
        - Token usage tracking for cost monitoring
        - Configurable timeouts and rate limiting
        
    Configuration options:
        - api_key: OpenAI API key or compatible service key
        - base_url: API endpoint (defaults to OpenAI, supports custom servers)
        - model_name: Target model identifier
        - max_tokens: Maximum response length
        - temperature: Response creativity (0.0-2.0)
        - embedding_dimensions: Custom embedding size (if supported)
        - organization: OpenAI organization ID
        - timeout: Request timeout in seconds
        - max_retries: Automatic retry attempts
    """
    
    @cached_property
    def client(self):
        """Get OpenAI client (cached)"""
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
            model_name = getattr(self.config, 'model_name', 'gpt-4o-mini')
            logger.info(f"OpenAI client initialized: {model_name}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    # ==================== CHAT IMPLEMENTATION ====================
    
    def _chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        """Internal chat implementation"""
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Message format error: must contain 'role' and 'content'")
        
        # Get config values
        model_name = getattr(self.config, 'model_name', 'gpt-4o-mini')
        default_max_tokens = getattr(self.config, 'max_tokens', 2000)
        default_temperature = getattr(self.config, 'temperature', 0.7)
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens or default_max_tokens,
                temperature=temperature or default_temperature,
                **kwargs
            )
            
            result = response.choices[0].message.content.strip()
            
            if return_token_count:
                token_stats = {
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                return result, token_stats
            else:
                return result
                
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ):
        """
        Streaming chat completion using OpenAI
        """
        self.validate_task_support('chat')
        return self._stream_chat(messages, max_tokens, temperature, return_token_count, **kwargs)
    
    def _stream_chat(
        self,
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ):
        """
        Internal streaming chat implementation
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Message format error: must contain 'role' and 'content'")
            if not self.validate_input(msg['content']):
                raise ValueError(f"Message content validation failed: {msg['content']}")
        
        try:
            model_name = getattr(self.config, 'model_name', 'gpt-4o-mini')
            default_max_tokens = getattr(self.config, 'max_tokens', 2000)
            default_temperature = getattr(self.config, 'temperature', 0.7)
            
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens or default_max_tokens,
                temperature=temperature or default_temperature,
                stream=True,
                **kwargs
            )
            
            full_response = ""
            
            for chunk in stream:
                # Check for content in choices
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content = delta.content
                        full_response += content
                        yield content
                
                # Check for usage information (in final chunk)
                if return_token_count and hasattr(chunk, 'usage') and chunk.usage is not None:
                    token_stats = {
                        "input_tokens": chunk.usage.prompt_tokens if chunk.usage else 0,
                        "output_tokens": chunk.usage.completion_tokens if chunk.usage else 0,
                        "total_tokens": chunk.usage.total_tokens if chunk.usage else 0
                    }
                    
                    yield token_stats
                    
        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            raise
    
    # ==================== EMBEDDING IMPLEMENTATION ====================
    
    def _embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using OpenAI
        """
        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        # Validate inputs
        for text in text_list:
            if not self.validate_input(text):
                raise ValueError(f"Text validation failed: {text}")
        
        try:
            # Clean texts - remove newlines
            cleaned_texts = [text.replace("\n", " ") for text in text_list]
            
            model_name = getattr(self.config, 'model_name', 'text-embedding-ada-002')
            embedding_dimensions = getattr(self.config, 'embedding_dimensions', None)
            
            # Create embedding request
            embedding_kwargs = {}
            if embedding_dimensions:
                embedding_kwargs['dimensions'] = embedding_dimensions
            
            response = self.client.embeddings.create(
                model=model_name,
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
    
    # ==================== CONVENIENCE METHODS ====================
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents - always returns list of embeddings"""
        return self._embed(texts) if isinstance(texts, list) else [self._embed(texts)]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query - always returns single embedding"""
        result = self._embed(text)
        return result if isinstance(result, list) and isinstance(result[0], (int, float)) else result[0]
    
    # ==================== UTILITY METHODS ====================
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        """
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            return model_names
        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed when getting available models: {e}")
            return []
        except openai.APIConnectionError as e:
            logger.error(f"API connection error when getting available models: {e}")
            return []
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded when getting available models: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error when getting available models: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        """
        info = super().get_model_info()
        
        # Safely get client info without forcing initialization
        client = getattr(self, 'client', None)
        
        info.update({
            "model": getattr(self.config, 'model_name', 'gpt-4o-mini'),
            "api_base": getattr(client, 'base_url', None) if client else None,
            "organization": getattr(client, 'organization', None) if client else None,
            "max_retries": getattr(client, 'max_retries', None) if client else None,
            "timeout": getattr(client, 'timeout', None) if client else None,
            "default_max_tokens": getattr(self.config, 'max_tokens', 2000),
            "default_temperature": getattr(self.config, 'temperature', 0.7),
            "embedding_dimensions": getattr(self.config, 'embedding_dimensions', None),
            "provider": "openai"
        })
        return info
    
    # ==================== NOT SUPPORTED ====================
    
    def _rerank(self, query: str, documents: List['Document'], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """OpenAI doesn't provide native reranking"""
        raise NotImplementedError("OpenAI provider does not support reranking. Use a dedicated reranker.")