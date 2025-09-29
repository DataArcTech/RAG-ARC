from typing import Literal, Optional, Annotated
from pydantic import Field

from framework.config import AbstractConfig
from core.file_management.parser_combinator import ParserCobinator
from config.core.file_management.parser.dots_ocr import DotsOCRConfig
from config.core.file_management.parser.native import NativeParserConfig


class ParserCobinatorConfig(AbstractConfig):
    """Configuration for ParserCobinator"""
    type: Literal["parser_combinator"] = "parser_combinator"
    parser: Optional[Annotated[DotsOCRConfig | NativeParserConfig, Field(discriminator="type")]] = None

    def build(self) -> ParserCobinator:
        return ParserCobinator(self)