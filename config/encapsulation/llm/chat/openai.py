"""Configuration for OpenAI Chat LLM"""

import os
from framework.config import AbstractConfig
from encapsulation.llm.chat.openai import OpenAIChatLLM
from typing import Literal, Optional


def _resolve_chat_provider():
    provider = os.getenv("CHAT_MODEL_PROVIDER", "openai").lower()
    return provider if provider in {"openai", "huggingface"} else "openai"


class OpenAIChatConfig(AbstractConfig):
    """Configuration for OpenAI Chat LLM"""
    # Discriminator for config type identification
    type: Literal["openai_chat"] = "openai_chat"

    # Loading method configuration - can choose between providers
    loading_method: Literal["openai", "huggingface"] = _resolve_chat_provider()  # Provider for model loading

    # Model configuration
    model_name: str = os.getenv("CHAT_MODEL_NAME", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))  # Chat model identifier
    max_tokens: int = 2000  # Maximum tokens in response
    temperature: float = 0.7  # Response creativity (0.0-2.0, higher = more creative)

    # API configuration - loaded from environment variables
    openai_api_key: str = os.getenv("CHAT_API_KEY", os.getenv("OPENAI_API_KEY", ""))  # API key for this chat provider
    openai_base_url: str = os.getenv("CHAT_API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))  # Endpoint for this chat provider
    organization: Optional[str] = None  # OpenAI organization ID (optional)

    # Connection configuration
    timeout: float = 60.0  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on failure

    def build(self) -> OpenAIChatLLM:
        return OpenAIChatLLM(self)
