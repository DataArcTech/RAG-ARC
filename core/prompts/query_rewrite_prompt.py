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

QUERY_REWRITE_INTENT_SYSTEM_SUFFIX: Final[str] = (
    "Additionally, you must output a JSON object ONLY (no markdown/code fences) with keys:\n"
    "- intent: string enum in {RAG_REQUIRED, CLARIFICATION, CORRECTION, CHITCHAT_ACK, TOPIC_SWITCH}\n"
    "- rewritten_query: string\n"
    "- anchors: array of strings (optional)\n"
    "  * anchors are the explicit company/product/entity names that the question refers to.\n"
    "  * IMPORTANT: anchors should be ATOMIC entity names (e.g. company name, product name), NOT full sentences.\n"
    "  * Keep each anchor short and likely to appear verbatim in document titles/filenames/headings.\n"
    "  * If the user uses a descriptive phrase, prefer the DISTINCTIVE name segment(s) that are most likely to appear\n"
    "    verbatim in filenames/headings (drop generic descriptors that may vary across documents).\n"
    "  * If a user mentions a long mixed-script name (e.g. Chinese + 'Upgrade' + Chinese), split it into multiple anchors\n"
    "    (e.g. the company name + the core product name segment) so filename matching remains effective.\n"
    "  * Prefer anchors that will match the corpus naming (e.g. if the corpus uses the short product name in filenames,\n"
    "    include that short name as an anchor even if the user used a longer marketing name).\n"
    "  * Prefer extracting anchors from the USER messages in the conversation context.\n"
    "  * If intent is TOPIC_SWITCH or CORRECTION, anchors MUST reflect the user's latest explicit subject.\n"
    "  * For CORRECTION: if the user mentions BOTH the intended subject and the mistaken/wrong subject,\n"
    "    anchors MUST include ONLY the intended subject (do NOT include the wrong one).\n"
    "- reason: short string (optional)\n"
    "\n"
    "Intent routing guidance (domain-agnostic):\n"
    "- CHITCHAT_ACK: greetings/thanks/acknowledgements with no info request AND no request to revise/explain.\n"
    "- CLARIFICATION: the user asks to revise/clarify/supplement/justify the assistant's previous answer;\n"
    "  or provides feedback about what should have been included; focus on alignment rather than new retrieval.\n"
    "- CORRECTION: the user corrects the subject/entity or says the assistant answered the wrong company/product.\n"
    "- TOPIC_SWITCH: the user explicitly switches to a different subject/product.\n"
    "- Otherwise: RAG_REQUIRED.\n"
    "\n"
    "Return STRICT JSON only."
)

QUERY_REWRITE_AND_INTENT_USER_PROMPT: Final[str] = (
    "Decide whether to rewrite this query for better retrieval AND classify the intent.\n"
    "Constraints:\n"
    "- Preserve the user query language (do NOT translate).\n"
    "- Preserve key numbers, product names, and proper nouns.\n"
    "- Do NOT broaden scope or add unrelated entities.\n"
    "- If the current query is a follow-up and omits the entity/product, and the conversation context includes the entity/product,\n"
    "  minimally disambiguate by injecting the relevant anchor(s) so retrieval does not drift.\n"
    "- anchors should list atomic entity names (company name and/or product name) used for disambiguation.\n"
    "- If the user explicitly corrects/switches the subject, do NOT inject old anchors; use the newest subject.\n"
    "\n"
    "User query: {query}\n"
    "\n"
    "Output JSON with keys: intent, rewritten_query, anchors, reason."
)

QUERY_REWRITE_AND_INTENT_USER_PROMPT_WITH_HISTORY: Final[str] = (
    QUERY_REWRITE_AND_INTENT_USER_PROMPT
    + "\n\nConversation context (most recent first, may be truncated):\n{history}"
)

__all__ = [
    "QUERY_REWRITE_USER_PROMPT",
    "QUERY_REWRITE_USER_PROMPT_WITH_HISTORY",
    "QUERY_REWRITE_ROUTING_SYSTEM_SUFFIX",
    "QUERY_REWRITE_AND_ROUTING_USER_PROMPT",
    "QUERY_REWRITE_AND_ROUTING_USER_PROMPT_WITH_HISTORY",
    "QUERY_REWRITE_INTENT_SYSTEM_SUFFIX",
    "QUERY_REWRITE_AND_INTENT_USER_PROMPT",
    "QUERY_REWRITE_AND_INTENT_USER_PROMPT_WITH_HISTORY",
]
