from typing import Literal, Optional, Annotated
from pydantic import Field

from framework.config import AbstractConfig
from core.file_management.parser.standard import StandardParser
from config.encapsulation.parser.dots_ocr import DotsOCRConfig
from config.encapsulation.parser.native import NativeParserConfig


class StandardParserConfig(AbstractConfig):
    """Configuration for StandardParser"""
    type: Literal["standard_parser"] = "standard_parser"
    parser: Optional[Annotated[DotsOCRConfig | NativeParserConfig, Field(discriminator="type")]] = None

    def build(self) -> StandardParser:
        return StandardParser(self)