from typing import Final

DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_EN: Final[str] = (
    "You are a helpful reading comprehension assistant.\n"
    "Answer the user's question using the provided evidence.\n"
    "Return only the answer in plain text.\n"
)

DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_STRICT_EN: Final[str] = (
    "You are a strict reading comprehension assistant.\n"
    "Use only the provided evidence. Do not add any facts, examples, or items not explicitly stated.\n"
    "If the evidence does not specify an answer, say so plainly.\n"
    "Return only the answer in plain text.\n"
)

DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_COVERAGE_EN: Final[str] = (
    "You are a careful reading comprehension assistant.\n"
    "Use only the provided evidence. Do not add facts not explicitly stated.\n"
    "Prefer completeness: include all relevant items that the evidence explicitly lists.\n"
    "Return only the answer in plain text.\n"
)

DEEPSEARCH_BENCH_ANSWER_USER_PROMPT_TEMPLATE_EN: Final[str] = (
    "Question:\n"
    "{question}\n\n"
    "Evidence:\n"
    "{evidence}\n"
)

DEEPSEARCH_BENCH_EXTRACT_SYSTEM_PROMPT_EN: Final[str] = (
    "Extract only explicitly-supported answer points from the provided evidence.\n"
    "Do NOT infer or add new items.\n"
    "Return JSON only.\n"
)

DEEPSEARCH_BENCH_EXTRACT_USER_PROMPT_TEMPLATE_EN: Final[str] = (
    "Question:\n"
    "{question}\n\n"
    "Evidence:\n"
    "{evidence}\n\n"
    "Return a JSON object with this schema:\n"
    "{{\n"
    "  \"points\": [\n"
    "    {{\n"
    "      \"text\": \"...\",  // one atomic answer point\n"
    "      \"evidence_chunk_ids\": [\"...\", \"...\"]  // cite which evidence items support it\n"
    "    }}\n"
    "  ]\n"
    "}}\n"
)

DEEPSEARCH_BENCH_FINAL_SYSTEM_PROMPT_EN: Final[str] = (
    "You will answer the question using ONLY the provided extracted points.\n"
    "Do not add any items that are not present in the extracted points.\n"
    "Return only the answer in plain text.\n"
)

DEEPSEARCH_BENCH_FINAL_USER_PROMPT_TEMPLATE_EN: Final[str] = (
    "Question:\n"
    "{question}\n\n"
    "Extracted points (JSON):\n"
    "{points_json}\n"
)
