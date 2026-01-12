"""Prompt templates for graph retrieval components (centralized)."""

FACT_RERANK_USER_PROMPT = (
    "User query:\n{query}\n\n"
    "Candidate facts (1-based indices):\n{facts_text}\n\n"
    "Task:\n"
    "Select the {k} most relevant facts.\n\n"
    "Return ONLY valid JSON:\n"
    '- A JSON object with a single key "indices" whose value is a JSON array of integers (1-based indices), '
    'e.g. {{"indices":[1,3,5]}}\n'
    "Constraints:\n"
    "- Do not include any extra text.\n"
    "- Only use indices that appear in the candidate list.\n"
)

FACT_RERANK_RETRY_USER_PROMPT = (
    "User query:\n{query}\n\n"
    "Candidate facts (1-based indices):\n{facts_text}\n\n"
    "Task:\n"
    "Select the {k} most relevant facts.\n\n"
    "Return EXACTLY one JSON object and nothing else:\n"
    '{{"indices":[1,3,5]}}\n'
    "Constraints:\n"
    "- Output must start with '{{' and end with '}}'.\n"
    "- No markdown fences.\n"
    "- No explanations.\n"
    "- Only use indices that appear in the candidate list.\n"
)

SEED_ENTITY_FILTER_USER_PROMPT = (
    "User query:\n{query}\n\n"
    "Entity candidates (1-based indices):\n{entities_text}\n\n"
    "Task:\n"
    "Select 2-5 most relevant entities as seed entities for graph traversal.\n\n"
    "Return ONLY valid JSON:\n"
    "- A JSON array of integers (1-based indices), e.g. [1, 2, 4]\n"
    "Constraints:\n"
    "- Do not include any extra text.\n"
    "- Only use indices that appear in the candidate list.\n"
)


__all__ = ["FACT_RERANK_USER_PROMPT", "FACT_RERANK_RETRY_USER_PROMPT", "SEED_ENTITY_FILTER_USER_PROMPT"]
