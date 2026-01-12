import os
from typing import Literal, Optional, Annotated
from pydantic import Field

from framework.config import AbstractConfig
from core.file_management.parser_combinator import ParserCombinator
from config.core.file_management.parser.dots_ocr import DotsOCRParserConfig
from config.core.file_management.parser.vlm_ocr import VLMOcrParserConfig
from config.core.file_management.parser.native import NativeParserConfig
from config.core.file_management.parser.mineru import MinerUParserConfig
from config.core.file_management.parser_mode import ParserParseMode, resolve_parser_parse_mode


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
    base_output_dir: str = Field(
        default_factory=lambda: str(os.getenv("PARSER_OUTPUT_DIR", "./data/parsed_files") or "./data/parsed_files"),
        description="Unified base output directory for all parsers.",
    )

    native_output_subdir: str = Field("native", description="Subdirectory name for native parser outputs.")
    dots_ocr_output_subdir: str = Field("dots_ocr", description="Subdirectory name for DotsOCR parser outputs.")
    vlm_ocr_output_subdir: str = Field("vlm_ocr", description="Subdirectory name for VLM OCR parser outputs.")
    mineru_output_subdir: str = Field("mineru", description="Subdirectory name for MinerU parser outputs.")

    parse_mode: ParserParseMode = Field(
        default_factory=resolve_parser_parse_mode,
        description="PDF/image parse mode (native|dotsocr|mineru).",
    )

    mineru_fallback_to_native_on_failure: bool = Field(
        default_factory=lambda: str(os.getenv("MINERU_FALLBACK_TO_NATIVE_ON_FAILURE", "true") or "true").strip().lower()
        in {"1", "true", "yes", "y", "on"},
        description=(
            "When PARSER_PARSE_MODE=mineru, fallback to native PDF text extraction if MinerU parsing fails "
            "(e.g., service not running). Fallback is observable via parse result metadata."
        ),
    )

    # OCR parser for PDF and images (optional)
    ocr_parser: Optional[Annotated[
        DotsOCRParserConfig | VLMOcrParserConfig | MinerUParserConfig,
        Field(discriminator="type")
    ]] = None

    # Optional named candidates (recommended when PARSER_PARSE_MODE drives runtime selection).
    dotsocr_parser: Optional[DotsOCRParserConfig] = None
    mineru_parser: Optional[MinerUParserConfig] = None

    # Native parser for office documents and text files (optional)
    native_parser: Optional[NativeParserConfig] = None

    concurrent_num: int = 20

    def build(self) -> ParserCombinator:
        resolved = self._resolve_for_build()
        return ParserCombinator(resolved)

    def _resolve_for_build(self) -> "ParserCombinatorConfig":
        # Env wins at runtime so operators can switch modes without editing json_configs.
        env_mode = str(os.getenv("PARSER_PARSE_MODE", "") or "").strip().lower()
        mode = env_mode if env_mode in {"native", "dotsocr", "mineru"} else self.parse_mode

        if mode == "native":
            return self.model_copy(update={"ocr_parser": None})

        if mode == "dotsocr":
            candidate = self.dotsocr_parser or self.ocr_parser
            if candidate is None or getattr(candidate, "type", None) != "dots_ocr_parser":
                raise ValueError(
                    "PARSER_PARSE_MODE=dotsocr requires DotsOCRParserConfig; "
                    "set parser_config.dotsocr_parser in knowledge config."
                )
            return self.model_copy(update={"ocr_parser": candidate})

        if mode == "mineru":
            candidate = self.mineru_parser or self.ocr_parser
            if candidate is None or getattr(candidate, "type", None) != "mineru_parser":
                raise ValueError(
                    "PARSER_PARSE_MODE=mineru requires MinerUParserConfig; "
                    "set parser_config.mineru_parser in knowledge config."
                )
            return self.model_copy(update={"ocr_parser": candidate})

        return self
