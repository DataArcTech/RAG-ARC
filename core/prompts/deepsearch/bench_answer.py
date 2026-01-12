from typing import Final

DEEPSEARCH_BENCH_ANSWER_SYSTEM_PROMPT_EN: Final[str] = (
    "You are a helpful reading comprehension assistant.\n"
    "Answer the user's question using the provided evidence.\n"
    "Return only the answer in plain text.\n"
)

DEEPSEARCH_BENCH_ANSWER_USER_PROMPT_TEMPLATE_EN: Final[str] = (
    "Question:\n"
    "{question}\n\n"
    "Evidence:\n"
    "{evidence}\n"
)

