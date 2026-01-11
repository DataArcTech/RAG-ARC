from core.retrieval.hybrid.rrf import RRFInput, fuse_rrf_with_scores


def test_fuse_rrf_with_bonus_prefers_both_sources() -> None:
    ranked_ids, scores = fuse_rrf_with_scores(
        sources={
            "bm25": [RRFInput(id="a", rank=1), RRFInput(id="b", rank=2)],
            "graph_neighborhood": [RRFInput(id="b", rank=1), RRFInput(id="c", rank=2)],
        },
        rrf_k=60,
        weights={"bm25": 1.0, "graph_neighborhood": 1.0},
        both_sources_bonus=1.0,
        k=3,
    )

    assert ranked_ids[0] == "b"
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]

