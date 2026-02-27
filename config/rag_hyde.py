"""HyDE (Hypothetical Document Embedding) configuration.

HyDE generates a hypothetical document passage from the user query,
then uses it alongside the original query for dense retrieval.
This improves recall by bridging the vocabulary gap between questions
and document passages.
"""
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def rag_hyde_enabled() -> bool:
    """Whether HyDE dual-query is enabled for dense retrieval (default: true).

    When enabled, the rewrite stage generates a hypothetical document passage
    in parallel with the standard query rewrite. The dense retriever then
    searches with both [original_query, hyde_query] and deduplicates via
    sum(score) / sqrt(n+1) aggregation.

    HyDE remains active even in bench_mode because it improves retrieval
    quality without confounding algorithm comparisons.
    """
    return _env_bool("RAG_HYDE_ENABLED", True)
