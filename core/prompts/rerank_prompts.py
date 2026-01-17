from typing import Final

LISTWISE_RERANK_DEFAULT_PROMPT_TEMPLATE: Final[str] = """The following documents are related to query: {QUERY}

Documents:
{DOC_STR}

Ranking guidelines:
- Treat company names, product names, and other proper nouns in the query as high-priority constraints.
- Prefer documents that explicitly mention the SAME company/product as the query.
- If the query specifies a company/product, documents about other companies/products should be ranked lower even if they share generic keywords (e.g., "multi-currency", "features", "discount").
- When there exists at least one document that explicitly mentions the query's company/product name(s), any document that does NOT mention them must be ranked lower than those that do.

Task:
- Rank the documents by relevance to the query.

Output requirements (STRICT):
- Output ONLY a JSON array of exactly {TOPK} UNIQUE integers.
- Integers are 1-based document ids (i.e., document [1] has id 1).
- Do NOT include any other text, explanation, markdown, or code fences.

Example:
[3, 1, 2]
"""
