"""Prompts for LLM-driven retrieval query variants."""

QUERY_VARIANTS_SYSTEM_PROMPT = (
    "You rewrite user search queries into specific target languages for retrieval.\n"
    "Never invent or substitute product/brand names. Keep proper nouns exactly as in the input.\n"
    "Keep rewrites short: do NOT add explanations, extra clauses, or reasoning.\n"
    "Return ONLY a JSON object. Do not add commentary."
)

QUERY_VARIANTS_USER_PROMPT_TEMPLATE = (
    "Rewrite the query into each target language. Keep product names, model numbers, and entities unchanged.\n"
    "Do NOT translate or substitute brand/product names. Do NOT introduce new names.\n"
    "Do NOT expand into long paragraphs; preserve the original meaning with minimal rewriting.\n"
    "If rewriting would change a name or you are unsure, reuse the original phrasing for that language.\n"
    "Output JSON only, with keys exactly matching target_langs.\n"
    "Payload:\n"
    "{payload}"
)

__all__ = ["QUERY_VARIANTS_SYSTEM_PROMPT", "QUERY_VARIANTS_USER_PROMPT_TEMPLATE"]
