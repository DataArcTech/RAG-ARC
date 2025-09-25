from typing import Literal, Optional
from framework.config import AbstractConfig
from encapsulation.parser.dots_ocr import DotsOCRParser


class DotsOCRConfig(AbstractConfig):
    """Configuration for DotsOCR Parser"""
    type: Literal["dots_ocr"] = "dots_ocr"

    # Model configuration
    # model_path: str = "rednote-hilab/dots.ocr"  # Default to HuggingFace model ID, can be local path
    model_path: str = "/finance_ML/dataarc_syn_database/model/rednote-hilab/DotsOCR"
    cache_dir: Optional[str] = None
    use_hf: bool = True
    device: str = "cuda:1"
    dpi: int = 200
    min_pixels: int = None
    max_pixels: int = None

    # VLLM configuration (when use_hf=False)
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "sk-xxx"
    model_name: str = "model"
    max_completion_tokens: int = 16384
    temperature: float = 0.1
    top_p: float = 1.0
    num_threads: int = 4

    def build(self) -> DotsOCRParser:
        return DotsOCRParser(self)