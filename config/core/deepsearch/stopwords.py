"""Stopword configuration for DeepSearch evidence ranking."""
from core.utils.stopwords import get_stopwords

_EVIDENCE_STOPWORDS_LANGS = ("en", "zh")

EVIDENCE_RANK_STOPWORDS = frozenset(
    token
    for lang in _EVIDENCE_STOPWORDS_LANGS
    for token in get_stopwords(lang)
)
