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
    "- If the query contains an important concept term that is likely to appear in source documents under a different but "
    "equivalent phrasing (synonym/alias), you SHOULD append up to 2 close equivalents in the SAME language to improve recall. "
    "Keep this minimal and do NOT introduce new intent or broaden the scope.\n"
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

QUERY_REWRITE_ROUTING_SYSTEM_SUFFIX: Final[str] = (
    "Additionally, you must output a JSON object ONLY (no markdown/code fences) with keys:\n"
    "- rewritten_query: string\n"
    "- retrieval_ratios: object with numeric fields {dense, bm25, graph}\n"
    "  * These ratios control how many retrieval candidates to allocate to each backend.\n"
    "  * Use higher graph ratio when the question likely needs multi-entity linking / relationship evidence across chunks/files.\n"
    "  * Use higher dense/bm25 ratios when the answer is likely a specific clause/number/threshold in one document.\n"
    "  * Keep ratios small, non-negative, and comparable (e.g. 1, 1, 1.5).\n"
    "- reason: short string (optional)\n"
    "\n"
    "Return STRICT JSON only."
)

QUERY_REWRITE_AND_ROUTING_USER_PROMPT: Final[str] = (
    "Decide whether to rewrite this query for better retrieval.\n"
    "Constraints:\n"
    "- If the user intent is already clear and the query is retrieval-ready AND no disambiguation/script-normalization is needed, "
    "keep rewritten_query equal to the ORIGINAL query verbatim.\n"
    "- Preserve the user query language (do NOT translate).\n"
    "- Preserve key numbers, product names, and proper nouns.\n"
    "- Do NOT expand the question into new subquestions or add generic terms that may drift intent.\n"
    "\n"
    "User query: {query}\n"
    "\n"
    "Output JSON with keys: rewritten_query, retrieval_ratios, reason."
)

QUERY_REWRITE_AND_ROUTING_USER_PROMPT_WITH_HISTORY: Final[str] = (
    QUERY_REWRITE_AND_ROUTING_USER_PROMPT
    + "\n\nConversation context (most recent first, may be truncated):\n{history}"
)

__all__ = [
    "QUERY_REWRITE_USER_PROMPT",
    "QUERY_REWRITE_USER_PROMPT_WITH_HISTORY",
    "QUERY_REWRITE_ROUTING_SYSTEM_SUFFIX",
    "QUERY_REWRITE_AND_ROUTING_USER_PROMPT",
    "QUERY_REWRITE_AND_ROUTING_USER_PROMPT_WITH_HISTORY",
]
