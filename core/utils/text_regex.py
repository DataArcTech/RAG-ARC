"""Shared regex patterns for text normalization/parsing.

Keep this module dependency-free so it can be reused across core/encapsulation/framework.
"""
import re

# Markdown fenced blocks (primarily used for LLM JSON outputs).
MARKDOWN_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", flags=re.DOTALL | re.IGNORECASE)

# Inline citation tokens used by DeepSearch reporters (e.g. [ev1] or 【ev1】).
INLINE_CITATION_TOKEN_RE = re.compile(
    r"(?:\[(?P<bracket>[^\[\]]{1,128})\]|【(?P<cjk>[^【】]{1,128})】)"
)
BRACKET_CONTENT_RE = re.compile(r"\[([^\[\]]+)\]")
CJK_BRACKET_CONTENT_RE = re.compile(r"【([^【】]+)】")

# Generic text utilities.
WHITESPACE_RE = re.compile(r"\s+")
NONWORD_SPACES_RE = re.compile(r"[\s\u3000]+")

# Common list/path detection (DeepSearch query cleanup).
BULLET_RE = re.compile(r"^\s*[-*]\s+")
PATH_LIKE_RE = re.compile(r"[/\\]|\.(pdf|docx|pptx|xlsx|md|txt)\b", flags=re.IGNORECASE)

# Sentence-level heuristics.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# CJK detection (broad enough for heuristic language detection).
CJK_DETECT_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
