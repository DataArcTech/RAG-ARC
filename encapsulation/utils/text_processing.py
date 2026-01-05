"""
Encapsulation-layer wrapper for text normalization.

We keep a single implementation in `core.utils.text_processing` to avoid drift.
"""

from core.utils.text_processing import normalize_entity_text, text_processing  # noqa: F401
