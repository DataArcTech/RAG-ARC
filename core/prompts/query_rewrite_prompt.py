"""Prompt templates for query rewriting.

Query rewriting is used for retrieval effectiveness only. It must not introduce
language drift: rewritten queries should stay in the same language as the user query.
"""

from typing import Final

QUERY_REWRITE_USER_PROMPT: Final[str] = (
    "Decide whether to rewrite this query for better retrieval.\n"
    "Constraints:\n"
    "- If the user intent is already clear and the query is retrieval-ready AND no disambiguation/script-normalization is needed, "
    "return the ORIGINAL query verbatim.\n"
    "- If the query refers to a specific product/entity but the name is abbreviated/partial, and the conversation context "
    "contains a more specific/full name, rewrite to use that fuller name (minimal disambiguation).\n"
    "- If key proper nouns (e.g., company/product names) appear in a different Chinese script (Simplified vs Traditional) "
    "than the likely source documents, you MAY add the alternate-script variant of those proper nouns ONLY (minimal change). "
    "If helpful, include both variants for the SAME proper noun.\n"
    "- If the current query is ambiguous (e.g., refers to a previously mentioned product/company as 'this plan'/'that product'), "
    "use the conversation context (if provided) to minimally disambiguate.\n"
    "- Preserve the user query language (do NOT translate).\n"
    "- Preserve key numbers, product names, and proper nouns.\n"
    "- Do NOT expand the question into new subquestions or add generic terms that may drift intent.\n"
    "- Return ONLY the rewritten query text.\n"
    "\n"
    "User query: {query}"
)

QUERY_REWRITE_USER_PROMPT_WITH_HISTORY: Final[str] = (
    QUERY_REWRITE_USER_PROMPT
    + "\n\nConversation context (most recent first, may be truncated):\n{history}"
)

__all__ = ["QUERY_REWRITE_USER_PROMPT", "QUERY_REWRITE_USER_PROMPT_WITH_HISTORY"]
