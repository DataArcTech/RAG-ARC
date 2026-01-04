import numpy as np

from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_ppr import _PrunedHippoRAGNeo4jPPRMixin


class _DummyPPR(_PrunedHippoRAGNeo4jPPRMixin):
    def __init__(self, *, directed_mode: str, directed_relations: set[str], backend: str = "push"):
        self.passage_node_keys = ["chunk-1"]
        self.ppr_backend = backend
        self.ppr_directed_mode = directed_mode
        self._directed_relations = directed_relations
        self.calls: list[tuple[str, object]] = []

    def _directionality_config(self) -> dict:
        return {
            "schema_loaded": True,
            "direction_policy": "whitelist",
            "directed_relations": set(self._directed_relations),
            "direction_insensitive_relations": set(),
        }

    def _run_ppr_push(self, subgraph_nodes, reset, damping, owner_id=None):  # noqa: ARG002
        self.calls.append(("push", None))
        return {"chunk-1": 0.9}

    def _run_ppr_igraph(  # noqa: PLR0913
        self,
        subgraph_nodes,
        reset,
        damping,
        owner_id=None,
        direction_policy: str = "whitelist",  # noqa: ARG002
        directed_relations=None,
        direction_insensitive_relations=None,  # noqa: ARG002
    ):  # noqa: ARG002
        self.calls.append(("igraph", set(directed_relations or set())))
        return {"chunk-1": 0.9}


def test_ppr_directed_mode_auto_prefers_directed_when_schema_declares_relations() -> None:
    dummy = _DummyPPR(directed_mode="auto", directed_relations={"OWNS"}, backend="push")
    doc_ids, doc_scores, scores = dummy._run_ppr_with_weights(
        node_weights={"chunk-1": 1.0},
        damping=0.5,
        subgraph_nodes={"chunk-1"},
    )
    assert scores.get("chunk-1") == 0.9
    assert dummy.calls[0][0] == "igraph"
    assert dummy.calls[0][1] == {"OWNS"}
    assert np.all(doc_ids == np.array([0]))
    assert np.all(doc_scores == np.array([0.9]))


def test_ppr_directed_mode_off_keeps_push_backend() -> None:
    dummy = _DummyPPR(directed_mode="off", directed_relations={"OWNS"}, backend="push")
    _, _, _ = dummy._run_ppr_with_weights(node_weights={"chunk-1": 1.0}, damping=0.5, subgraph_nodes={"chunk-1"})
    assert dummy.calls[0][0] == "push"


def test_ppr_directed_mode_auto_without_sensitive_relations_keeps_backend() -> None:
    dummy = _DummyPPR(directed_mode="auto", directed_relations=set(), backend="push")
    _, _, _ = dummy._run_ppr_with_weights(node_weights={"chunk-1": 1.0}, damping=0.5, subgraph_nodes={"chunk-1"})
    assert dummy.calls[0][0] == "push"
