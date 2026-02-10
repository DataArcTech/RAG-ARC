from core.knowledge_graph.maintenance.gradient_select import GradientSelect, truncate_indices


def test_truncate_indices_keeps_mink_then_gradient_drop() -> None:
    scores = [1.0, 0.9, 0.81, 0.70, 0.69]
    keep = truncate_indices(scores, policy=GradientSelect(mink=1, g=0.9))
    assert keep == [0, 1, 2]


def test_truncate_indices_mink_zero_keeps_first_item() -> None:
    scores = [0.5, 0.49, 0.1]
    keep = truncate_indices(scores, policy=GradientSelect(mink=0, g=0.6))
    assert keep[:1] == [0]

