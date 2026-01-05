from typing import Literal, Annotated, Optional, Any
from pydantic import Field
from framework.config import AbstractConfig
from core.file_management.parser.dots_ocr import DotsOCRParser
from config.encapsulation.llm.parse.dots_ocr import DotsOCRConfig


class DotsOCRParserConfig(AbstractConfig):
    """Configuration for DotsOCR Parser (Core Layer)"""
    type: Literal["dots_ocr_parser"] = "dots_ocr_parser"

    output_dir: Optional[str] = Field(
        default=None,
        description="Output directory for OCR artifacts. When using ParserCombinator this is set automatically.",
    )

    # LLM service configuration (required)
    dots_ocr: DotsOCRConfig

    # Parsing configuration
    dpi: int = 200  # DPI for PDF page conversion
    min_pixels: Optional[int] = None  # Minimum image pixels
    max_pixels: Optional[int] = 4000000  # Maximum image pixels (default: 4000000 ≈ 2000x2000, recommended to avoid GPU OOM)
    num_threads: int = 1 # Thread count for PDF processing

    # Default parsing parameters
    default_prompt_mode: str = "prompt_layout_all_en"  # Default prompt mode
    default_bbox: Optional[Any] = None  # Default bounding box
    default_fitz_preprocess: bool = False  # Default fitz preprocessing

    def build(self) -> DotsOCRParser:
        return DotsOCRParser(self)
