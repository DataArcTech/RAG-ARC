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

def _default_low_cost_model_name() -> Optional[str]:
    value = os.getenv("LOW_COST_MODEL")
    token = (value or "").strip()
    return token or None


def _default_api_key() -> str:
    return os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY", "")


def _default_base_url() -> str:
    return os.getenv("CHAT_API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))


def _default_timeout() -> float:
    raw = os.getenv("CHAT_TIMEOUT_SECONDS", "")
    token = str(raw or "").strip()
    if token:
        try:
            return float(token)
        except ValueError:
            pass
    return 60.0


class OpenAIChatConfig(AbstractConfig):
    """Configuration for OpenAI Chat LLM"""

    type: Literal["openai_chat"] = "openai_chat"

    loading_method: Literal["openai", "huggingface"] = Field(
        default_factory=_resolve_chat_provider
    )
    model_name: str = Field(default_factory=_default_model_name)
    low_cost_model_name: Optional[str] = Field(
        default_factory=_default_low_cost_model_name,
        description="Optional cheaper model used for exploration-heavy calls (planning/reflection/quality checks).",
    )
    max_tokens: int = 2000
    temperature: float = 0.1

    # HuggingFace-only knobs (used when loading_method="huggingface")
    device: str = Field(default_factory=lambda: os.getenv("CHAT_MODEL_DEVICE", os.getenv("DEVICE", "cpu")))
    cache_folder: Optional[str] = Field(default_factory=lambda: os.getenv("CHAT_MODEL_CACHE_FOLDER"))
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    openai_api_key: str = Field(default_factory=_default_api_key)
    openai_base_url: str = Field(default_factory=_default_base_url)
    organization: Optional[str] = None
    timeout: float = Field(default_factory=_default_timeout)
    max_retries: int = 3

    def build(self) -> OpenAIChatLLM:
        if self.loading_method == "openai":
            if not str(self.openai_api_key or "").strip():
                raise ValueError("Chat API key is required: set CHAT_API_KEY or OPENAI_API_KEY.")
            if not str(self.openai_base_url or "").strip():
                raise ValueError("Chat base URL is required: set CHAT_API_BASE_URL or OPENAI_BASE_URL.")
        return OpenAIChatLLM(self)
