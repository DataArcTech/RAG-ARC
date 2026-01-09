from types import SimpleNamespace

from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_ppr import _PrunedHippoRAGNeo4jPPRMixin


def test_ppr_push_uses_configured_epsilon() -> None:
    captured = {}

    class _GraphStore:
        def compute_ppr_push(  # noqa: ANN001
            self,
            *,
            subgraph_nodes,
            reset,
            alpha,
            epsilon,
            push_threshold_mode,
            target_degree_penalty_gamma,
            owner_id,
        ):
            captured.update(
                {
                    "subgraph_nodes": subgraph_nodes,
                    "reset": reset,
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "push_threshold_mode": push_threshold_mode,
                    "target_degree_penalty_gamma": target_degree_penalty_gamma,
                    "owner_id": owner_id,
                }
            )
            return {"chunk-x": 0.1}

    class _Runner(_PrunedHippoRAGNeo4jPPRMixin):
        def __init__(self):
            self.graph_store = _GraphStore()
            self.config = SimpleNamespace(
                ppr_push_epsilon=0.000123,
                ppr_push_threshold_mode="residual",
                ppr_push_target_degree_penalty_gamma=0.5,
            )

        @staticmethod
        def _owner_to_str(owner_id):  # noqa: ANN001
            return owner_id

        def _run_ppr_igraph(self, *args, **kwargs):  # noqa: ANN001,ARG002
            raise AssertionError("should not fall back in this test")

    runner = _Runner()
    out = runner._run_ppr_push(subgraph_nodes={"n1"}, reset={"n1": 1.0}, damping=0.5, owner_id=None)
    assert out == {"chunk-x": 0.1}
    assert captured["epsilon"] == 0.000123
    assert captured["push_threshold_mode"] == "residual"
    assert captured["target_degree_penalty_gamma"] == 0.5
