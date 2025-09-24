from typing import Literal
from framework.config import AbstractConfig
from encapsulation.parser.native import NativeParser


class NativeParserConfig(AbstractConfig):
    """Configuration for Native Parser"""
    type: Literal["native"] = "native"

    def build(self) -> NativeParser:
        return NativeParser(self)