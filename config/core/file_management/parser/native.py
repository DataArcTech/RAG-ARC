import os
from typing import Literal, Optional
from pydantic import Field
from framework.config import AbstractConfig
from core.file_management.parser.native import NativeParser


def _default_native_output_dir() -> Optional[str]:
    value = str(os.getenv("NATIVE_PARSER_OUTPUT_DIR", "") or "").strip()
    return value or None


class NativeParserConfig(AbstractConfig):
    """Configuration for Native Parser (Core Layer)"""
    type: Literal["native_parser"] = "native_parser"

    output_dir: Optional[str] = Field(
        default_factory=_default_native_output_dir,
        description="Output directory for native parser artifacts. When using ParserCombinator this is set automatically.",
    )

    def build(self) -> NativeParser:
        return NativeParser(self)
