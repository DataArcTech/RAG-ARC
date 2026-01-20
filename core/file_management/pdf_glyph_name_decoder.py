import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PdfGlyphDecodeResult:
    text: str
    changed: bool


# When a PDF lacks a usable ToUnicode cmap, extractors may emit glyph *names*
# like "/one.lf" (lining figure). These are not user-visible content and should
# be normalized for downstream chunking / retrieval.
_DIGIT_GLYPH_PREFIX = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

# Match digit glyph names with the common ".lf" suffix.
# We intentionally allow trailing characters (e.g. "/one.lf-" or "/one.lfPB-")
# because some PDFs surface these names concatenated with punctuation.
_LINING_FIGURE_NAME_RE = re.compile(
    r"/(?P<digit>zero|one|two|three|four|five|six|seven|eight|nine)\.lf",
    flags=re.IGNORECASE,
)


def decode_pypdf_glyph_names(text: str) -> PdfGlyphDecodeResult:
    """
    Decode common PDF glyph-name artifacts (e.g. "/one.lf") into their expected
    Unicode characters.

    This is specifically to mitigate broken native PDF text extraction where the
    extractor returns glyph names instead of actual characters.
    """
    if not isinstance(text, str) or not text or "/" not in text:
        return PdfGlyphDecodeResult(text=str(text or ""), changed=False)

    changed = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal changed
        token = str(match.group("digit") or "").lower()
        mapped = _DIGIT_GLYPH_PREFIX.get(token)
        if mapped is None:
            return match.group(0)
        changed = True
        return mapped

    decoded = _LINING_FIGURE_NAME_RE.sub(_repl, text)
    return PdfGlyphDecodeResult(text=decoded, changed=changed)

