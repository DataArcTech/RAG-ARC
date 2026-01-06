"""Language policy helpers for DeepSearch.

We need a robust "user language" signal that is not polluted by:
- file paths / bullet file lists
- document titles embedded in brackets (e.g. 《...》)

This signal is used to enforce output language across planner/report stages.
"""
import re
from typing import Literal

from core.deepsearch.utils.query_clean import clean_query
from core.utils.text_regex import CJK_DETECT_RE, PATH_LIKE_RE

UserLanguage = Literal["en", "zh"]

_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
_CJK_TITLE_BLOCK_RE = re.compile(r"《[^》]{1,160}》")


def normalize_for_language_detection(text: str) -> str:
    """Reduce a user question to the parts most indicative of the user's language."""

    value = clean_query(str(text or "").strip(), max_chars=480)
    if not value:
        return ""

    # Remove inline file/path-like tokens (e.g. ".../a.pdf") that can carry cross-language characters.
    # Keep non-path text around them intact.
    raw_tokens = value.split()
    tokens: list[str] = []
    for idx, raw in enumerate(raw_tokens):
        if PATH_LIKE_RE.search(raw):
            continue
        # Heuristic: when a filename token is split by whitespace, drop the immediately preceding
        # token if it is CJK-heavy and directly adjacent to a path-like token (e.g. "... （英文版）.pdf").
        next_token = raw_tokens[idx + 1] if idx + 1 < len(raw_tokens) else ""
        if (
            next_token
            and PATH_LIKE_RE.search(next_token)
            and CJK_DETECT_RE.search(raw)
            and not _ASCII_ALPHA_RE.search(raw)
        ):
            continue
        tokens.append(raw)
    value = " ".join(tokens).strip()

    # Remove CJK-title blocks that are often document names injected into an otherwise English question.
    value = _CJK_TITLE_BLOCK_RE.sub("", value).strip()
    return value


def infer_user_language(question: str) -> UserLanguage:
    """Infer the user language from the question (English vs Chinese).

    This intentionally prioritizes user-authored query text over referenced file names.
    """

    text = normalize_for_language_detection(question)
    if not text:
        return "en"

    alpha = len(_ASCII_ALPHA_RE.findall(text))
    cjk = len(CJK_DETECT_RE.findall(text))

    # Heuristic: if there is substantial ASCII alpha and it dominates, treat as English.
    if alpha >= 6 and alpha >= max(1, cjk) * 2:
        return "en"
    if cjk >= 2 and cjk >= alpha:
        return "zh"
    # Tie-breaker: any CJK defaults to Chinese, otherwise English.
    return "zh" if CJK_DETECT_RE.search(text) else "en"


def output_language_label(lang: UserLanguage) -> str:
    if lang == "zh":
        return "Chinese (Simplified)"
    return "English"
