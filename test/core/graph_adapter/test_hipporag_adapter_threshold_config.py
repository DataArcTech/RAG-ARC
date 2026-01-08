from types import SimpleNamespace

from config.core.deepsearch.hipporag_adapter_defaults import HIPPORAG_ADAPTER_SEMANTIC_SCORE_THRESHOLD
from core.graph_adapter import registry


def test_hipporag_adapter_semantic_score_threshold_is_configurable() -> None:
    retriever = SimpleNamespace()
    adapter = registry.build_adapter("hipporag", retriever=retriever, semantic_score_threshold=0.12)
    assert adapter.semantic_score_threshold == 0.12


def test_hipporag_adapter_semantic_score_threshold_default_is_centralized() -> None:
    retriever = SimpleNamespace()
    adapter = registry.build_adapter("hipporag", retriever=retriever)
    assert adapter.semantic_score_threshold == HIPPORAG_ADAPTER_SEMANTIC_SCORE_THRESHOLD

