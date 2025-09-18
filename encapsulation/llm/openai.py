from encapsulation.llm.base import LLMBase, LLMBaseConfig
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING, Literal
from pydantic import Field
import openai
from dataclasses import dataclass
from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from core.utils.data_model import Document

class OpenAIConfig(LLMBaseConfig):
    """
    OpenAI LLM configuration
    """
    type: Literal["openai"] = "openai"
    task_types: Literal["chat", "embedding"] = Field(default="chat", description="Supported task types")

    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    base_url: Optional[str] = Field(default=None, description="API base URL")
    organization: Optional[str] = Field(default=None, description="Organization ID")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout: float = Field(default=60.0, description="Request timeout")
    default_max_tokens: Optional[int] = Field(default=None, description="Default max tokens for chat")
    default_temperature: float = Field(default=0.7, description="Default temperature for chat")
    
    def build(self) -> "OpenAILLM":
        """Build the OpenAI LLM"""
        return OpenAILLM(config=self)


@shared_module
class OpenAILLM(LLMBase[OpenAIConfig]):
    """
    Unified OpenAI LLM supporting both chat and embeddings
    Single client, multiple capabilities
    """

    config: OpenAIConfig

    def __init__(self, config: OpenAIConfig):
        """Initialize OpenAI LLM with config"""
        self.config = config

        # Setup logging first
        self._setup_logging()

        # Single OpenAI client for all operations
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization=self.config.organization,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout
        )

        self.async_client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization=self.config.organization,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout
        )

        self.logger.info(f"OpenAI LLM initialized: {self.config.model_name}")
    
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
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
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
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
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
                    return

                if return_token_count:
                    self.logger.info("The streaming chat has not finished yet, token usage information is temporarily unavailable")
                    default_stats = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    }
                    yield default_stats
                    
        except Exception as e:
            self.logger.error(f"Streaming chat failed: {str(e)}")
            raise


    async def achat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None, 
        return_token_count: bool = False, 
        **kwargs
        ) -> Union[str, Tuple[str, Dict[str, int]]]:
        """
        Async chat completion
        """
        self.validate_task_support('chat')
        return await self._achat(messages, max_tokens, temperature, return_token_count, **kwargs)

    async def astream_chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None, 
        return_token_count: bool = False, 
        **kwargs
        ):
        self.validate_task_support('chat')
        return await self._astream_chat(messages, max_tokens, temperature, return_token_count, **kwargs)

    async def _achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, int]]]:

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
                **kwargs
            )
            result = response.choices[0].message.content
            if return_token_count:
                return result, self._get_token_stats(response.usage)
            return result

        except Exception as e:
            self.logger.error(f"异步对话生成失败: {str(e)}")
            raise

    async def _astream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ):

        try:
            params = {}
            if return_token_count:
                params["stream_options"] = {"include_usage": True}

            stream = await self.async_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
                stream=True,
                **params,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

                if return_token_count and getattr(chunk, "usage", None):
                    yield self._get_token_stats(chunk.usage)

        except Exception as e:
            self.logger.error(f"异步流式对话生成失败: {str(e)}")
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
            
            
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=cleaned_texts,
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
            "model": self.config.model_name,
            "api_base": getattr(self.client, 'base_url', None),
            "organization": getattr(self.client, 'organization', None),
            "max_retries": getattr(self.client, 'max_retries', None),
            "timeout": getattr(self.client, 'timeout', None),
            "default_max_tokens": self.config.default_max_tokens,
            "default_temperature": self.config.default_temperature,
            "provider": "openai"
        })
        return info
    
    # ==================== NOT SUPPORTED ====================
    
    def _rerank(self, query: str, documents: List['Document'], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """OpenAI doesn't provide native reranking"""
        raise NotImplementedError("OpenAI provider does not support reranking. Use a dedicated reranker.")