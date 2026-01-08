from encapsulation.database.utils.ppr_push import ppr_push


def test_ppr_push_target_degree_penalty_reduces_hub_attraction() -> None:
    leaves = [f"leaf-{i}" for i in range(5)]
    adjacency = {
        "hub": [("s", 1.0), *[(leaf, 1.0) for leaf in leaves]],
        "s": [("hub", 1.0), ("tail", 1.0)],
        "tail": [("s", 1.0)],
    }
    for leaf in leaves:
        adjacency[leaf] = [("hub", 1.0)]

    reset = {"s": 1.0}

    scores_no_penalty = ppr_push(
        adjacency=adjacency,
        reset=reset,
        alpha=0.5,
        epsilon=1e-10,
        push_threshold_mode="residual",
        target_degree_penalty_gamma=0.0,
        max_iterations=10000,
    )
    scores_with_penalty = ppr_push(
        adjacency=adjacency,
        reset=reset,
        alpha=0.5,
        epsilon=1e-10,
        push_threshold_mode="residual",
        target_degree_penalty_gamma=1.0,
        max_iterations=10000,
    )

    assert scores_with_penalty.get("hub", 0.0) < scores_no_penalty.get("hub", 0.0)
    assert scores_with_penalty.get("tail", 0.0) > scores_no_penalty.get("tail", 0.0)

