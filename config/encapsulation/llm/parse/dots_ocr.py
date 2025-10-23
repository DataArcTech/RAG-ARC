from typing import Literal, Optional
from framework.config import AbstractConfig
from encapsulation.llm.parse.dots_ocr import DotsOCRLLM


class DotsOCRConfig(AbstractConfig):
    """Configuration for DotsOCR LLM Service"""
    type: Literal["dots_ocr"] = "dots_ocr"

    # Loading method configuration
    loading_method: Literal["huggingface", "vllm"] = "huggingface"
    use_china_mirror: bool = False
    cache_folder: Optional[str] = None
    use_snapshot_download: bool = False  # Use snapshot_download to avoid dynamic module issues

    # Model configuration
    device: str = "cuda:0"

    # HuggingFace configuration

    model_path: str = "rednote-hilab/dots.ocr"

    # VLLM configuration (when loading_method="vllm")
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "sk-xxx"
    model_name: str = "model"
    max_completion_tokens: int = 16384
    temperature: float = 0.1
    top_p: float = 1.0

    def build(self) -> DotsOCRLLM:
        return DotsOCRLLM(self)