from typing import Final

LISTWISE_RERANK_DEFAULT_PROMPT_TEMPLATE: Final[str] = """The following documents are related to query: {QUERY}

Documents:
{DOC_STR}

First identify the essential problem in the query. Think step by step to reason about why each document is relevant or irrelevant. Rank these documents based on their relevance to the query.
Please output the ranking result of documents as a list, where the first element is the id of the most relevant document, the second element is the id of the second most element, etc.
Please strictly follow the format to output a list of {TOPK} ids corresponding to the most relevant {TOPK} documents, sorted from the most to least relevant document. First think step by step and write the reasoning process, then output the ranking results as a list of ids in a json format like
```json
[... integer ids here ...]
```
"""

