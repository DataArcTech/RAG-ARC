import numpy as np

from core.retrieval.graph_retrieveal.fact_groundability import (
    FactGroundabilityConfig,
    apply_groundability,
    compute_overlap,
)


class _FakeChunk:
    def __init__(self, metadata: dict):
        self.metadata = metadata


def test_compute_overlap_missing_provenance_returns_none_ratio():
    overlap_count, overlap_ratio = compute_overlap(source_chunk_ids=[], retrieved_chunk_ids={"c1"})
    assert overlap_count == 0
    assert overlap_ratio is None


def test_apply_groundability_hard_filter_drops_ungrounded_and_penalizes_missing():
    scores = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    fact_ids = ["f1", "f2", "f3"]
    docstore = {
        "f1": _FakeChunk({"source_chunk_ids": ["c1", "c2"]}),
        "f2": _FakeChunk({"source_chunk_ids": ["c3"]}),
        "f3": _FakeChunk({}),
    }
    cfg = FactGroundabilityConfig(
        enabled=True,
        mode="hard_filter",
        dense_top_k=30,
        min_overlap_count=1,
        min_overlap_ratio=0.0,
        soft_min_weight=0.2,
        soft_gamma=1.0,
        keep_missing_provenance=True,
        missing_provenance_weight=0.2,
    )

    new_scores, new_fact_ids, meta = apply_groundability(
        cfg=cfg,
        scores=scores,
        fact_ids=fact_ids,
        docstore=docstore,
        dense_top_chunk_ids={"c2"},
    )

    assert new_fact_ids == ["f1", "f3"]
    assert np.allclose(new_scores, np.array([1.0, 0.16], dtype=np.float32))
    assert meta["mode"] == "hard_filter"
    assert meta["kept"] == 2
    assert meta["dropped"] == 1
    assert meta["missing_provenance"] == 1


def test_apply_groundability_soft_penalty_keeps_all_and_downweights():
    scores = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    fact_ids = ["f1", "f2", "f3"]
    docstore = {
        "f1": _FakeChunk({"source_chunk_ids": ["c1", "c2"]}),
        "f2": _FakeChunk({"source_chunk_ids": ["c3"]}),
        "f3": _FakeChunk({}),
    }
    cfg = FactGroundabilityConfig(
        enabled=True,
        mode="soft_penalty",
        dense_top_k=30,
        min_overlap_count=1,
        min_overlap_ratio=0.0,
        soft_min_weight=0.2,
        soft_gamma=1.0,
        keep_missing_provenance=True,
        missing_provenance_weight=0.2,
    )

    new_scores, new_fact_ids, meta = apply_groundability(
        cfg=cfg,
        scores=scores,
        fact_ids=fact_ids,
        docstore=docstore,
        dense_top_chunk_ids={"c2"},
    )

    assert new_fact_ids == ["f1", "f2", "f3"]
    assert np.allclose(new_scores, np.array([0.6, 0.18, 0.16], dtype=np.float32))
    assert meta["mode"] == "soft_penalty"
    assert meta["kept"] == 3
    assert meta["dropped"] == 0
    assert meta["missing_provenance"] == 1


def test_apply_groundability_uses_fallback_provenance_mapping():
    scores = np.array([1.0, 0.9], dtype=np.float32)
    fact_ids = ["f1", "f2"]
    docstore = {
        "f1": _FakeChunk({}),  # Missing provenance in docstore
        "f2": _FakeChunk({"source_chunk_ids": ["c9"]}),
    }
    cfg = FactGroundabilityConfig(
        enabled=True,
        mode="hard_filter",
        dense_top_k=30,
        min_overlap_count=1,
        min_overlap_ratio=0.0,
        soft_min_weight=0.2,
        soft_gamma=1.0,
        keep_missing_provenance=False,
        missing_provenance_weight=0.2,
    )

    new_scores, new_fact_ids, meta = apply_groundability(
        cfg=cfg,
        scores=scores,
        fact_ids=fact_ids,
        docstore=docstore,
        dense_top_chunk_ids={"c1"},
        fallback_source_chunk_ids_by_fact_id={"f1": ["c1"]},
    )

    assert new_fact_ids == ["f1"]
    assert np.allclose(new_scores, np.array([1.0], dtype=np.float32))
    assert meta["missing_provenance"] == 0
