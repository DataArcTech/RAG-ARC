import os
from typing import Literal, Optional
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.llm.parse.dots_ocr import DotsOCRLLM


class DotsOCRConfig(AbstractConfig):
    """Configuration for DotsOCR LLM Service"""
    type: Literal["dots_ocr"] = "dots_ocr"

    # Loading method configuration
    loading_method: Literal["huggingface", "vllm"] = Field(
        default_factory=lambda: os.getenv("DOTS_OCR_LOADING_METHOD", "huggingface").lower()
    )
    use_china_mirror: bool = Field(
        default_factory=lambda: os.getenv("DOTS_OCR_USE_CHINA_MIRROR", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    cache_folder: Optional[str] = Field(default_factory=lambda: os.getenv("DOTS_OCR_CACHE_FOLDER", "./models/dots_ocr"))
    use_snapshot_download: bool = Field(
        default_factory=lambda: os.getenv("DOTS_OCR_USE_SNAPSHOT_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    )  # Use snapshot_download to avoid dynamic module issues

    # Model configuration
    device: str = Field(default_factory=lambda: os.getenv("DOTS_OCR_DEVICE", os.getenv("DEVICE", "cpu")))

    # HuggingFace configuration
    model_path: str = Field(default_factory=lambda: os.getenv("DOTS_OCR_MODEL_PATH", "rednote-hilab/dots.ocr"))

    # VLLM configuration (when loading_method="vllm")
    base_url: str = Field(default_factory=lambda: os.getenv("DOTS_OCR_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = Field(default_factory=lambda: os.getenv("DOTS_OCR_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.getenv("DOTS_OCR_VLLM_MODEL_NAME", "model"))
    max_completion_tokens: int = Field(default_factory=lambda: int(os.getenv("DOTS_OCR_MAX_COMPLETION_TOKENS", "16384")))
    temperature: float = Field(default_factory=lambda: float(os.getenv("DOTS_OCR_TEMPERATURE", "0.1")))
    top_p: float = Field(default_factory=lambda: float(os.getenv("DOTS_OCR_TOP_P", "1.0")))

    def build(self) -> DotsOCRLLM:
        return DotsOCRLLM(self)
