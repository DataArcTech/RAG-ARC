"""Prompt templates for query rewriting.

Query rewriting is used for retrieval effectiveness only. It must not introduce
language drift: rewritten queries should stay in the same language as the user query.
"""

from typing import Final

QUERY_REWRITE_USER_PROMPT: Final[str] = (
    "Decide whether to rewrite this query for better retrieval.\n"
    "Constraints:\n"
    "- If the user intent is already clear and the query is retrieval-ready, return the ORIGINAL query verbatim.\n"
    "- Preserve the user query language (do NOT translate).\n"
    "- Preserve key numbers, product names, and proper nouns.\n"
    "- Do NOT expand the question into new subquestions or add generic terms that may drift intent.\n"
    "- Return ONLY the rewritten query text.\n"
    "\n"
    "User query: {query}"
)
