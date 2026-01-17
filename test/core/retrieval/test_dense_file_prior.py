from core.retrieval.graph_retrieveal.dense_file_prior import (
    compute_dense_file_prior_multipliers,
    dense_file_distribution_stats,
    should_apply_dense_file_prior,
)


def test_dense_file_distribution_stats_top_and_second():
    ids = ["a", "b", "a", "c", "a", "b"]
    stats = dense_file_distribution_stats(ids, top_k=6)
    assert stats["total"] == 6
    assert stats["unique"] == 3
    assert stats["top_file_id"] == "a"
    assert stats["top_count"] == 3
    assert abs(stats["top_ratio"] - 0.5) < 1e-9
    assert stats["second_file_id"] == "b"
    assert stats["second_count"] == 2
    assert abs(stats["second_ratio"] - (2 / 6)) < 1e-9


def test_should_apply_dense_file_prior_requires_ratio_and_margin():
    assert should_apply_dense_file_prior(top_ratio=0.6, second_ratio=0.1, min_ratio=0.25, min_margin=0.1)
    # ratio ok but margin too small
    assert not should_apply_dense_file_prior(top_ratio=0.32, second_ratio=0.30, min_ratio=0.25, min_margin=0.1)
    # margin ok but ratio too small
    assert not should_apply_dense_file_prior(top_ratio=0.2, second_ratio=0.0, min_ratio=0.25, min_margin=0.1)


def test_compute_dense_file_prior_multipliers_supports_multiple_files():
    ids = ["a", "b", "a", "b"]
    multipliers, stats = compute_dense_file_prior_multipliers(
        ids,
        top_k=4,
        max_files=2,
        min_ratio=0.25,
        min_margin=0.0,
        multiplier=3.0,
    )
    assert stats["applied"] is True
    # Both files are equally dominant, so both get the full multiplier.
    assert multipliers == {"a": 3.0, "b": 3.0}
