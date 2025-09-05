from .base import LLMBase
from typing import Dict, Any, List, Optional, Union, Tuple
import openai


class OpenAILLM(LLMBase):
    """
    Unified OpenAI LLM supporting both chat and embeddings
    Single client, multiple capabilities
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        task_types: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize OpenAI LLM
        
        Args:
            model_name: Model name (gpt-4, gpt-3.5-turbo, text-embedding-3-small, etc.)
            api_key: OpenAI API key
            base_url: API base URL
            organization: Organization ID
            max_retries: Max retry attempts
            timeout: Request timeout
            task_types: Supported tasks ['chat', 'embedding'] or subset
            **kwargs: Additional config
        """
        # Default to chat support, but allow override
        task_types = task_types or ['chat']
        
        super().__init__(
            model_name=model_name, 
            task_types=task_types, 
            **kwargs
        )
        
        # Single OpenAI client for all operations
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Default parameters for chat
        self.default_max_tokens = kwargs.get('max_tokens', 2000)
        self.default_temperature = kwargs.get('temperature', 0.7)
        
        # Default parameters for embeddings
        self.embedding_dimensions = kwargs.get('embedding_dimensions', None)
        
        self.logger.info(f"OpenAI LLM initialized: {model_name}")
    
    # ==================== CHAT IMPLEMENTATION ====================
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        """
        Chat completion using OpenAI
        """
        self.validate_task_support('chat')
        return self._chat(messages, max_tokens, temperature, return_token_count, **kwargs)
    
    def _chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        """
        Internal chat implementation
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                **kwargs
            )
            
            result = response.choices[0].message.content.strip()
            
            if return_token_count:
                token_stats = {
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                
                self.logger.debug(f"Chat completion successful, length: {len(result)}, tokens: {token_stats}")
                return result, token_stats
            else:
                self.logger.debug(f"Chat completion successful, length: {len(result)}")
                return result
                
        except Exception as e:
            self.logger.error(f"Chat completion failed: {str(e)}")
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
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
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
                    
                    self.logger.debug(f"Streaming chat completed, length: {len(full_response)}, tokens: {token_stats}")
                    yield token_stats
                    
        except Exception as e:
            self.logger.error(f"Streaming chat failed: {str(e)}")
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
            
            self.logger.debug(f"Embedding successful, {len(embeddings)} vectors generated")
            
            # Return single embedding or list based on input
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding failed: {str(e)}")
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
            self.logger.debug(f"Retrieved {len(model_names)} available models")
            return model_names
        except Exception as e:
            self.logger.error(f"Failed to get model list: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        """
        info = super().get_model_info()
        info.update({
            "model": self.model_name,
            "api_base": getattr(self.client, 'base_url', None),
            "organization": getattr(self.client, 'organization', None),
            "max_retries": getattr(self.client, 'max_retries', None),
            "timeout": getattr(self.client, 'timeout', None),
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature,
            "embedding_dimensions": self.embedding_dimensions,
            "provider": "openai"
        })
        return info
    
    # ==================== NOT SUPPORTED ====================
    
    def _rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """OpenAI doesn't provide native reranking"""
        raise NotImplementedError("OpenAI provider does not support reranking. Use a dedicated reranker.")