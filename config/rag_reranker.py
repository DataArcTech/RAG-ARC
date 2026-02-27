"""Reranker type configuration.

Controls which reranker backend is used in the normal RAG pipeline.
"""
import os


def rag_reranker_type() -> str:
    """Return the configured reranker type.

    Values:
        'listwise' (default) — DashScope qwen-rerank cross-encoder API.
        'llm'                — Listwise reranking using the chat LLM model.

    Set via env var ``RAG_RERANKER_TYPE``.
    """
    return os.getenv("RAG_RERANKER_TYPE", "listwise").strip().lower()
