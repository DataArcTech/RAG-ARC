"""Shared helper to resolve the active graph store instance."""
from typing import Any

from framework.register import Register
from application.rag_inference.module import RAGInference

_registrator = Register()


def get_graph_store():
    """Return the graph store from the registered RAG inference module."""
    try:
        rag_module: RAGInference = _registrator.get_object("rag_inference")
    except KeyError:
        return None
    store = rag_module.get_graph_store()
    if store is None:
        return None
    return store
