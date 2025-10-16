"""Configuration for OpenAI Chat LLM"""

import os
from framework.config import AbstractConfig
from encapsulation.llm.chat.openai import OpenAIChatLLM
from typing import Literal, Optional


class OpenAIChatConfig(AbstractConfig):
    """Configuration for OpenAI Chat LLM"""
    # Discriminator for config type identification
    type: Literal["openai_chat"] = "openai_chat"

    # Loading method configuration - can choose between providers
    loading_method: Literal["openai", "huggingface"] = "openai"  # Provider for model loading

    # Model configuration
    model_name: str = "gpt-4o-mini"  # OpenAI model name (gpt-4, gpt-4o-mini, gpt-3.5-turbo, etc.)
    max_tokens: int = 2000  # Maximum tokens in response
    temperature: float = 0.7  # Response creativity (0.0-2.0, higher = more creative)

    # API configuration - loaded from environment variables
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")  # API key for authentication
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # API endpoint URL
    organization: Optional[str] = None  # OpenAI organization ID (optional)

    # Connection configuration
    timeout: float = 60.0  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on failure

    def build(self) -> OpenAIChatLLM:
        return OpenAIChatLLM(self)