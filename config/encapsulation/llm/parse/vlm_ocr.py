import os
from typing import Literal, Optional
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.llm.parse.vlm_ocr import VLMOcrLLM


def _resolve_ocr_provider():
    provider = os.getenv("OCR_MODEL_PROVIDER", "openai").lower()
    return provider if provider in {"openai", "vllm"} else "openai"


def _default_ocr_model_name() -> str:
    return os.getenv("OCR_MODEL_NAME", os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini"))


def _default_ocr_api_key() -> str:
    return os.getenv("OCR_API_KEY", os.getenv("OPENAI_API_KEY", ""))


def _default_ocr_base_url() -> str:
    return os.getenv("OCR_API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))


class VLMOcrConfig(AbstractConfig):
    """Configuration for VLM OCR LLM Service"""
    type: Literal["vlm_ocr"] = "vlm_ocr"

    # Loading method configuration
    loading_method: Literal["openai", "vllm"] = Field(default_factory=_resolve_ocr_provider)

    # Model configuration
    model_name: str = Field(default_factory=_default_ocr_model_name)

    # OpenAI configuration (when loading_method="openai")
    openai_api_key: str = Field(default_factory=_default_ocr_api_key)
    openai_base_url: str = Field(default_factory=_default_ocr_base_url)
    organization: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3

    # VLLM configuration (when loading_method="vllm")
    base_url: str = Field(default_factory=lambda: os.getenv("OCR_API_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")))
    api_key: str = Field(default_factory=_default_ocr_api_key)

    # Inference parameters
    max_tokens: int = 4096
    max_completion_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0

    def build(self) -> VLMOcrLLM:
        if self.loading_method == "openai":
            if not str(self.openai_api_key or "").strip():
                raise ValueError("OCR API key is required: set OCR_API_KEY or OPENAI_API_KEY.")
            if not str(self.openai_base_url or "").strip():
                raise ValueError("OCR base URL is required: set OCR_API_BASE_URL or OPENAI_BASE_URL.")
        return VLMOcrLLM(self)
