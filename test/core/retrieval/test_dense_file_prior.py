from core.retrieval.graph_retrieveal.dense_file_prior import pick_dense_top_file_id


def test_pick_dense_top_file_id_ignores_none_and_computes_ratio():
    file_id, ratio, count = pick_dense_top_file_id(
        [None, "a", "a", "b", "a", None],
        top_k=6,
    )
    assert file_id == "a"
    assert count == 3
    assert ratio == 3 / 4


def test_pick_dense_top_file_id_empty_returns_none():
    file_id, ratio, count = pick_dense_top_file_id([None, None], top_k=2)
    assert file_id is None
    assert ratio == 0.0
    assert count == 0

