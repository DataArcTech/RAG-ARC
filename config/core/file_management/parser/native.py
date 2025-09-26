from typing import Literal
from framework.config import AbstractConfig
from core.file_management.parser.native import NativeParser


class NativeParserConfig(AbstractConfig):
    """Configuration for Native Parser (Core Layer)"""
    type: Literal["native_parser"] = "native_parser"

    # No additional configuration needed for native parser
    # It handles DOCX, Excel, PowerPoint, HTML using built-in libraries

    def build(self) -> NativeParser:
        return NativeParser(self)