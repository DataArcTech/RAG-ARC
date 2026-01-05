import os
from typing import Literal


ParserParseMode = Literal["native", "dotsocr", "mineru"]


def resolve_parser_parse_mode() -> ParserParseMode:
    """
    Resolve parsing mode for PDF/images.

    - native: extract text from PDF without OCR (similar to pdftotext); images unsupported.
    - dotsocr: local DotsOCR model (OCR).
    - mineru: remote MinerU parsing service (OCR + layout).
    """
    raw = str(os.getenv("PARSER_PARSE_MODE", "native") or "").strip().lower()
    return raw if raw in {"native", "dotsocr", "mineru"} else "native"

