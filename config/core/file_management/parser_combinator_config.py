from typing import Literal, Optional, Annotated
from pydantic import Field

from framework.config import AbstractConfig
from core.file_management.parser_combinator import ParserCombinator
from config.core.file_management.parser.dots_ocr import DotsOCRParserConfig
from config.core.file_management.parser.vlm_ocr import VLMOcrParserConfig
from config.core.file_management.parser.native import NativeParserConfig


class ParserCombinatorConfig(AbstractConfig):
    """
    Configuration for ParserCombinator.

    ParserCombinator combines OCR parser and Native parser to handle different file types:
    - OCR Parser (DotsOCR/VLM): For PDF and image files (.pdf, .jpg, .jpeg, .png)
    - Native Parser: For office documents and text files (.docx, .xlsx, .pptx, .html, .txt, etc.)

    At least one parser must be configured. If both are configured, the combinator will
    automatically select the appropriate parser based on file extension.

    Output Directory Structure:
        All parsers output to a unified base directory with subdirectories:
        - {base_output_dir}/native/     - Native parser outputs
        - {base_output_dir}/dots_ocr/   - DotsOCR parser outputs
        - {base_output_dir}/vlm_ocr/    - VLM OCR parser outputs

    Example:
        ```python
        config = ParserCombinatorConfig(
            base_output_dir="./data/parsed_files",
            ocr_parser=DotsOCRParserConfig(...),
            native_parser=NativeParserConfig()
        )
        ```
    """
    type: Literal["parser_combinator"] = "parser_combinator"

    # Unified base output directory for all parsers
    base_output_dir: str = Field("./data/parsed_files", description="Unified base output directory for all parsers.")

    native_output_subdir: str = Field("native", description="Subdirectory name for native parser outputs.")
    dots_ocr_output_subdir: str = Field("dots_ocr", description="Subdirectory name for DotsOCR parser outputs.")
    vlm_ocr_output_subdir: str = Field("vlm_ocr", description="Subdirectory name for VLM OCR parser outputs.")

    # OCR parser for PDF and images (optional)
    ocr_parser: Optional[Annotated[
        DotsOCRParserConfig | VLMOcrParserConfig,
        Field(discriminator="type")
    ]] = None

    # Native parser for office documents and text files (optional)
    native_parser: Optional[NativeParserConfig] = None

    concurrent_num: int = 20

    def build(self) -> ParserCombinator:
        return ParserCombinator(self)
