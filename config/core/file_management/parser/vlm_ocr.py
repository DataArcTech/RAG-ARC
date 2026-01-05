import os
from typing import Literal, Optional
from pydantic import Field
from framework.config import AbstractConfig
from core.file_management.parser.vlm_ocr import VLMOcrParser
from config.encapsulation.llm.parse.vlm_ocr import VLMOcrConfig


def _default_vlmocr_output_dir() -> Optional[str]:
    value = str(os.getenv("VLMOCR_OUTPUT_DIR", "") or "").strip()
    return value or None


class VLMOcrParserConfig(AbstractConfig):
    """Configuration for VLM OCR Parser (Core Layer) - Simple OCR Mode"""
    type: Literal["vlm_ocr_parser"] = "vlm_ocr_parser"

    output_dir: Optional[str] = Field(
        default_factory=_default_vlmocr_output_dir,
        description="Output directory for OCR artifacts. When using ParserCombinator this is set automatically.",
    )

    # LLM service configuration (required)
    vlm_ocr: VLMOcrConfig

    # Parsing configuration
    dpi: int = 200  # DPI for PDF page conversion
    num_threads: int = 1  # Thread count for PDF processing

    def build(self) -> VLMOcrParser:
        return VLMOcrParser(self)
