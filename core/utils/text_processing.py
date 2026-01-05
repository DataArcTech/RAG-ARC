import re

# Keep ASCII letters/digits, CJK Unified Ideographs, and spaces. Replace the rest with spaces.
_TEXT_NORMALIZATION_PATTERN = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff ]")


def text_processing(text: object) -> str:
    """
    Normalize text by:
    - converting to lowercase
    - removing punctuation/symbols (keeps alnum/CJK/spaces)
    - trimming surrounding whitespace
    """
    value = str(text) if text is not None else ""
    return _TEXT_NORMALIZATION_PATTERN.sub(" ", value.lower()).strip()


def normalize_entity_text(text: object) -> str:
    """Alias for `text_processing` kept for backward compatibility."""
    return text_processing(text)
