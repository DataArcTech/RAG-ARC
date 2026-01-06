"""Prompt templates for query rewriting.

Query rewriting is used for retrieval effectiveness only. It must not introduce
language drift: rewritten queries should stay in the same language as the user query.
"""

from typing import Final

QUERY_REWRITE_USER_PROMPT: Final[str] = (
    "Rewrite this query for better retrieval.\n"
    "Constraints:\n"
    "- Preserve the user query language (do NOT translate).\n"
    "- Preserve key numbers, product names, and proper nouns.\n"
    "- Return ONLY the rewritten query text.\n"
    "\n"
    "User query: {query}"
)

