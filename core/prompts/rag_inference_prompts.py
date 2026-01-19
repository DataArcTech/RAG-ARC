from typing import Final

#
# RAG chat system prompt layers (service-agnostic base + rag_inference-only addons).
#
# Motivation:
# - CLI is for debugging: it should share the same retrieval/rerank config, but keep prompting minimal.
# - rag_inference is user-facing: it needs stricter citation/format rules (and optional domain/style layers).
#
# NOTE: Keep prompts centralized (no scattered strings in business code).

RAG_CHAT_BASE_SYSTEM_PROMPT_EN: Final[str] = (
    "You are a helpful assistant.\n"
    "Answer in the same language as the user's question.\n"
    "Be concise and avoid making up facts.\n"
)

RAG_CHAT_CITATION_ADDON_SYSTEM_PROMPT_EN: Final[str] = (
    "You may be given a list of numbered Sources (key=1..N).\n"
    "Rules:\n"
    "1) If the user message is just a greeting / test / acknowledgement (e.g. '测试', 'test', 'hello', 'hi', '你好'),\n"
    "   answer briefly and DO NOT use any Sources and DO NOT include any <sup> tags.\n"
    "2) If Sources are provided (the list is not empty), ground your answer in Sources and add inline citations using HTML <sup> tags.\n"
    "   - Every sentence that contains factual information supported by Sources MUST end with one or more <sup>key</sup>.\n"
    "   - For MULTIPLE citations: Use SEPARATE <sup> tags with a single space between them.\n"
    "     Format: \"内容。<sup>1</sup> <sup>3</sup>\" (CORRECT), NOT \"内容。<sup>1,3</sup>\".\n"
    "   - Cite only the minimal number of sources needed; do NOT cite all sources by default.\n"
    "   - Do NOT output a bare block/list of citations without nearby supporting text.\n"
    "   - Do NOT cite a source you did not use.\n"
    "3) If NO Sources are provided (the list is empty), DO NOT use any <sup> tags in your answer.\n"
    "   - Say you don't know or cannot answer based on the available information.\n"
    "   - Do NOT make up citations or use <sup> tags when there are no Sources.\n"
    "4) If Sources are provided but none are relevant, say you don't know based on the provided Sources and ask a clarifying question.\n"
    "5) Do NOT use bracket citations like [1] and do NOT add a trailing 'Sources:' section.\n"
    "6) Output in Markdown. The only HTML allowed is <sup>...</sup>.\n"
)

# Backwards-compatible monolithic prompt used by legacy configs/YAML overrides.
RAG_INFERENCE_CITATION_SYSTEM_PROMPT_EN: Final[str] = (
    RAG_CHAT_BASE_SYSTEM_PROMPT_EN + "\n" + RAG_CHAT_CITATION_ADDON_SYSTEM_PROMPT_EN
)

RAG_INFERENCE_BENCH_SYSTEM_PROMPT_EN: Final[str] = (
    "You are a helpful reading comprehension assistant.\n"
    "Answer the user's question using the provided sources if they are available.\n"
    "Return only the answer in plain text.\n"
)
