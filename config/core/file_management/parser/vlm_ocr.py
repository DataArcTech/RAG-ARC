from typing import Literal
from framework.config import AbstractConfig
from core.file_management.parser.vlm_ocr import VLMOcrParser
from config.encapsulation.llm.parse.vlm_ocr import VLMOcrConfig


class VLMOcrParserConfig(AbstractConfig):
    """Configuration for VLM OCR Parser (Core Layer) - Simple OCR Mode"""
    type: Literal["vlm_ocr_parser"] = "vlm_ocr_parser"

    # LLM service configuration (required)
    vlm_ocr: VLMOcrConfig

    # Parsing configuration
    dpi: int = 200  # DPI for PDF page conversion
    num_threads: int = 1  # Thread count for PDF processing

    def build(self) -> VLMOcrParser:
        return VLMOcrParser(self)

