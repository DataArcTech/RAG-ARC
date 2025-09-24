"""Configuration for OpenAI Chat LLM"""

from framework.config import AbstractConfig
from encapsulation.llm.chat.openai import OpenAIChatLLM
from typing import Literal, Optional


class OpenAIChatConfig(AbstractConfig):
    """Configuration for OpenAI Chat LLM"""
    # Discriminator for config type identification
    type: Literal["openai_chat"] = "openai_chat"

    # Model configuration
    model_name: str = "gpt-4o-mini"  # OpenAI model name (gpt-4, gpt-4o-mini, gpt-3.5-turbo, etc.)
    max_tokens: int = 2000  # Maximum tokens in response
    temperature: float = 0.7  # Response creativity (0.0-2.0, higher = more creative)

    # API configuration
    base_url: str = "https://api.gptsapi.net/v1"  # API endpoint URL
    api_key: str = "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2"  # API key for authentication
    organization: Optional[str] = None  # OpenAI organization ID (optional)

    # Connection configuration
    timeout: float = 60.0  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on failure

    def build(self) -> OpenAIChatLLM:
        return OpenAIChatLLM(self)