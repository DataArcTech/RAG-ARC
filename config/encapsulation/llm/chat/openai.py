"""Configuration for OpenAI Chat LLM"""

import os
from typing import Any, Dict, Literal, Optional

from pydantic import Field

from encapsulation.llm.chat.openai import OpenAIChatLLM
from framework.config import AbstractConfig


def _resolve_chat_provider() -> str:
    provider = os.getenv("CHAT_MODEL_PROVIDER", "openai").lower()
    return provider if provider in {"openai", "huggingface"} else "openai"


def _default_model_name() -> str:
    return os.getenv("CHAT_MODEL_NAME", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))


def _default_api_key() -> str:
    return os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY", "")


def _default_base_url() -> str:
    return os.getenv("CHAT_API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))


class OpenAIChatConfig(AbstractConfig):
    """Configuration for OpenAI Chat LLM"""

    type: Literal["openai_chat"] = "openai_chat"

    loading_method: Literal["openai", "huggingface"] = Field(
        default_factory=_resolve_chat_provider
    )
    model_name: str = Field(default_factory=_default_model_name)
    max_tokens: int = 2000
    temperature: float = 0.7

    # HuggingFace-only knobs (used when loading_method="huggingface")
    device: str = Field(default_factory=lambda: os.getenv("CHAT_MODEL_DEVICE", os.getenv("DEVICE", "cpu")))
    cache_folder: Optional[str] = Field(default_factory=lambda: os.getenv("CHAT_MODEL_CACHE_FOLDER"))
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    openai_api_key: str = Field(default_factory=_default_api_key)
    openai_base_url: str = Field(default_factory=_default_base_url)
    organization: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3

    def build(self) -> OpenAIChatLLM:
        return OpenAIChatLLM(self)
