"""Shared helper to resolve the active graph store instance."""
from typing import Any, Optional

from framework.register import Register
from application.rag_inference.module import RAGInference

_registrator = Register()
_GRAPH_STORE: Optional[Any] = None


def get_graph_store():
    """Return the cached graph store from the registered RAG inference module."""

    global _GRAPH_STORE
    if _GRAPH_STORE is not None:
        return _GRAPH_STORE
    try:
        rag_module: RAGInference = _registrator.get_object("rag_inference")
    except KeyError:
        return None
    store = rag_module.get_graph_store()
    if store is None:
        return None
    _GRAPH_STORE = store
    return store
