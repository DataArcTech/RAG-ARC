from .base import ChatLLMBase
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from encapsulation.llm.utils.openai_client import create_openai_sync_client, create_openai_async_client
from encapsulation.llm.utils.huggingface_client import create_transformers_client
import logging

logger = logging.getLogger(__name__)


class OpenAIChatLLM(ChatLLMBase):
    """
    OpenAI Chat LLM implementation supporting conversational AI capabilities.

    This class provides a complete chat solution using OpenAI's API,
    supporting multiple model types with streaming and async capabilities.
    Optimized for conversational AI with configurable parameters and retry logic.

    Key features:
    - Multiple model support: GPT-4, GPT-3.5, GPT-4o variants
    - Streaming chat with token usage tracking
    - Async support for high-throughput applications
    - Configurable parameters: temperature, max_tokens
    - Automatic retry logic and timeout handling
    - Flexible API endpoint configuration (OpenAI or compatible)

    Configuration options:
        - api_key: OpenAI API key or compatible service key
        - base_url: API endpoint (defaults to OpenAI, supports custom servers)
        - model_name: Target model identifier
        - max_tokens: Maximum response length
        - temperature: Response creativity (0.0-2.0)
        - organization: OpenAI organization ID
        - timeout: Request timeout in seconds
        - max_retries: Automatic retry attempts
    """

    def __init__(self, config: "OpenAIChatConfig"):
        """Initialize OpenAI Chat with loading method support"""
        super().__init__(config)
        # Cache config values to avoid repeated getattr calls
        self.model_name = getattr(self.config, 'model_name', 'gpt-4o-mini')
        self.max_tokens = getattr(self.config, 'max_tokens', 2000)
        self.temperature = getattr(self.config, 'temperature', 0.7)
        self.loading_method = getattr(self.config, 'loading_method', 'openai')

        # Initialize client based on loading method
        if self.loading_method == 'openai':
            self.client = create_openai_sync_client(self.config)
            self.async_client = create_openai_async_client(self.config)
        elif self.loading_method == 'huggingface':
            # For HuggingFace transformers, we get (model, tokenizer) tuple
            self.client, self.tokenizer = create_transformers_client(self.config)
            self.async_client = None  # HuggingFace uses asyncio.to_thread wrapper
        else:
            raise ValueError(f"Unsupported loading method: {self.loading_method}")

    # ==================== CHAT IMPLEMENTATION ====================

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Internal chat implementation"""
        self._validate_messages(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )

            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """
        Internal streaming chat implementation
        """
        self._validate_messages(messages)

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                # Check for content in choices
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content = delta.content
                        yield content

        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            raise

    async def achat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Async function of chat
        """
        self._validate_messages(messages)

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            result = response.choices[0].message.content
            return result

        except Exception as e:
            logger.error(f"Async chat failed: {str(e)}")
            raise

    async def astream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Async function of stream chat
        """
        self._validate_messages(messages)

        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Async streaming chat failed: {str(e)}")
            raise

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
            "default_max_tokens": self.max_tokens,
            "default_temperature": self.temperature,
            "provider": "openai",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown')
        }

    def _validate_messages(self, messages):
        """Validate message format"""
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Message format error: must contain 'role' and 'content'")
            if not self._validate_input(msg['content']):
                raise ValueError(f"Message content validation failed: {msg['content']}")

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