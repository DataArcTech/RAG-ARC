import os


def rag_retrieval_dynamic_quota_enabled() -> bool:
    """
    Whether to enable LLM-driven, per-query routing ratios for MultiPath candidate quotas.

    Default: enabled.
    """
    raw = str(os.getenv("RAG_RETRIEVAL_DYNAMIC_QUOTA_ENABLED", "true") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}

