from typing import Final

LISTWISE_RERANK_DEFAULT_PROMPT_TEMPLATE: Final[str] = """The following documents are related to query: {QUERY}

Documents:
{DOC_STR}

Ranking guidelines:
- Treat company names, product names, and other proper nouns in the query as high-priority constraints.
- Prefer documents that explicitly mention the SAME company/product as the query.
- If the query specifies a company/product, documents about other companies/products should be ranked lower even if they share generic keywords (e.g., "multi-currency", "features", "discount").
- When there exists at least one document that explicitly mentions the query's company/product name(s), any document that does NOT mention them must be ranked lower than those that do.

First identify the essential problem in the query. Think step by step to reason about why each document is relevant or irrelevant. Rank these documents based on their relevance to the query.
Please output the ranking result of documents as a list, where the first element is the id of the most relevant document, the second element is the id of the second most element, etc.
Please strictly follow the format to output a list of exactly {TOPK} UNIQUE integer ids corresponding to the most relevant {TOPK} documents, sorted from the most to least relevant document.
If you are unsure about some positions, still output your best guess to reach exactly {TOPK} ids (do NOT output fewer than {TOPK}).
First think step by step and write the reasoning process, then output the ranking results as a list of ids in a json format like
```json
[... integer ids here ...]
```
"""
