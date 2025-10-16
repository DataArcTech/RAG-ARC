from typing import Literal, Optional
from framework.config import AbstractConfig
from encapsulation.llm.parse.vlm_ocr import VLMOcrLLM


class VLMOcrConfig(AbstractConfig):
    """Configuration for VLM OCR LLM Service"""
    type: Literal["vlm_ocr"] = "vlm_ocr"

    # Loading method configuration
    loading_method: Literal["openai", "vllm"] = "openai"

    # Model configuration
    model_name: str = "gpt-4o-mini"

    # OpenAI configuration (when loading_method="openai")
    openai_api_key: str
    openai_base_url: str
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

