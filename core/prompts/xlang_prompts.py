from typing import Final

FILE_SCOPE_XLANG_REWRITE_PROMPT_TEMPLATE: Final[str] = """You are helping a retrieval system bridge language gaps.
Translate the query and return STRICT JSON only.
Schema: {{"zh_hans": string, "zh_hant": string, "en": string}}.
Keep it concise; preserve numbers and key terms.

Query: {query}
"""


def build_file_scope_xlang_rewrite_prompt(*, query: str) -> str:
    return FILE_SCOPE_XLANG_REWRITE_PROMPT_TEMPLATE.format(query=str(query or "").strip())

