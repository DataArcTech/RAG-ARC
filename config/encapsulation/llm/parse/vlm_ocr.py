import os
from typing import Literal, Optional
from framework.config import AbstractConfig
from encapsulation.llm.parse.vlm_ocr import VLMOcrLLM


def _resolve_ocr_provider():
    provider = os.getenv("OCR_MODEL_PROVIDER", "openai").lower()
    return provider if provider in {"openai", "vllm"} else "openai"


class VLMOcrConfig(AbstractConfig):
    """Configuration for VLM OCR LLM Service"""
    type: Literal["vlm_ocr"] = "vlm_ocr"

    # Loading method configuration
    loading_method: Literal["openai", "vllm"] = _resolve_ocr_provider()

    # Model configuration
    model_name: str = os.getenv("OCR_MODEL_NAME", os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini"))

    # OpenAI configuration (when loading_method="openai")
    openai_api_key: str = os.getenv("OCR_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = os.getenv("OCR_API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
    organization: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3

    # VLLM configuration (when loading_method="vllm")
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "sk-xxx"

    # Inference parameters
    max_tokens: int = 4096
    max_completion_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0

    def build(self) -> VLMOcrLLM:
        return VLMOcrLLM(self)
