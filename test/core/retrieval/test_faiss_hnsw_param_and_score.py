from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from core.utils.retrieval_helper import RetrievalHelper
from encapsulation.data_model.schema import Chunk
from encapsulation.database.vector_db.faiss import FaissVectorDB


class _StubEmbeddingConfig:
    def __init__(self, *, dim: int = 2):
        self.type = "stub_embedding"
        self.loading_method = "stub"
        self.model_name = "stub"
        self.embedding_dimensions = int(dim)

    def build(self):  # noqa: D401
        # The FAISS DB will not call the embedder in this test because we provide precomputed embeddings.
        return object()


@dataclass
class _StubFaissConfig:
    embedding_config: _StubEmbeddingConfig
    index_type: str
    index_path: str

    metric: str = "cosine"
    normalize_L2: bool = True
    nlist: int = 32
    m: int = 12
    efConstruction: int = 20
    efSearch: int = 7
    train_size: int = 10000
    two_stage_enabled: bool = False
    two_stage_prefetch_k: int = 200


def _unit(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr) + 1e-12)
    return arr.astype(np.float32).tolist()


def test_hnsw_uses_config_params_and_scores_match_cosine(tmp_path: Path) -> None:
    # `FaissVectorDB` is a @shared_module; clear cache to avoid cross-test reuse.
    FaissVectorDB.clear_cache()

    storage = tmp_path / "hnsw_idx"
    cfg = _StubFaissConfig(
        embedding_config=_StubEmbeddingConfig(dim=2),
        index_type="hnsw",
        index_path=str(storage),
        m=12,
        efConstruction=20,
        efSearch=7,
    )
    db = FaissVectorDB(cfg)  # type: ignore[arg-type]

    v1 = _unit([1.0, 0.0])
    v2 = _unit([0.0, 1.0])
    v3 = _unit([1.0, 1.0])
    chunks = [
        Chunk(id="c1", content="a", owner_id="o1", metadata={"embedding": v1, "owner_id": "o1"}),
        Chunk(id="c2", content="b", owner_id="o1", metadata={"embedding": v2, "owner_id": "o1"}),
        Chunk(id="c3", content="c", owner_id="o1", metadata={"embedding": v3, "owner_id": "o1"}),
    ]
    db.update_index(chunks)

    # Index parameters should be applied (no hardcoded defaults).
    assert db.index is not None
    assert db.index.hnsw.efConstruction == 20
    assert db.index.hnsw.efSearch == 7
    # nb_neighbors(1) equals M for level>=1.
    assert db.index.hnsw.nb_neighbors(1) == 12

    q = _unit([1.0, 0.0])
    results = RetrievalHelper.vector_search_with_faiss(db, q, {"k": 3, "metric": "cosine"})
    assert [c.id for c, _s in results][:3] == ["c1", "c3", "c2"]

    # Scores should align with cosine similarity.
    score_map = {c.id: float(s) for c, s in results}
    assert score_map["c1"] == pytest.approx(1.0, abs=1e-4)
    assert score_map["c3"] == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-3)
    assert score_map["c2"] == pytest.approx(0.0, abs=1e-3)

    # Search-time should enforce efSearch from config (if the index is re-used across calls).
    assert db.index.hnsw.efSearch == 7


def test_flat_does_not_use_hnsw_score_conversion(tmp_path: Path) -> None:
    FaissVectorDB.clear_cache()

    storage = tmp_path / "flat_idx"
    cfg = _StubFaissConfig(
        embedding_config=_StubEmbeddingConfig(dim=2),
        index_type="flat",
        index_path=str(storage),
    )
    db = FaissVectorDB(cfg)  # type: ignore[arg-type]
    chunks = [
        Chunk(id="c1", content="a", owner_id="o1", metadata={"embedding": _unit([1.0, 0.0]), "owner_id": "o1"}),
        Chunk(id="c2", content="b", owner_id="o1", metadata={"embedding": _unit([0.0, 1.0]), "owner_id": "o1"}),
    ]
    db.update_index(chunks)
    q = _unit([1.0, 0.0])
    results = RetrievalHelper.vector_search_with_faiss(db, q, {"k": 2, "metric": "cosine"})
    assert results[0][0].id == "c1"
    assert float(results[0][1]) == pytest.approx(1.0, abs=1e-4)


def test_hnsw_two_stage_prefetch_then_exact_rescore(tmp_path: Path) -> None:
    FaissVectorDB.clear_cache()

    storage = tmp_path / "hnsw_2stage_idx"
    cfg = _StubFaissConfig(
        embedding_config=_StubEmbeddingConfig(dim=2),
        index_type="hnsw",
        index_path=str(storage),
        two_stage_enabled=True,
        two_stage_prefetch_k=100,
        m=12,
        efConstruction=20,
        efSearch=7,
    )
    db = FaissVectorDB(cfg)  # type: ignore[arg-type]

    v1 = _unit([1.0, 0.0])
    v2 = _unit([0.0, 1.0])
    v3 = _unit([1.0, 1.0])
    chunks = [
        Chunk(id="c1", content="a", owner_id="o1", metadata={"embedding": v1, "owner_id": "o1"}),
        Chunk(id="c2", content="b", owner_id="o1", metadata={"embedding": v2, "owner_id": "o1"}),
        Chunk(id="c3", content="c", owner_id="o1", metadata={"embedding": v3, "owner_id": "o1"}),
    ]
    db.update_index(chunks)

    q = _unit([1.0, 0.0])
    results = RetrievalHelper.vector_search_with_faiss(db, q, {"k": 2, "metric": "cosine"})
    assert len(results) == 2
    assert [c.id for c, _s in results] == ["c1", "c3"]
    score_map = {c.id: float(s) for c, s in results}
    assert score_map["c1"] == pytest.approx(1.0, abs=1e-4)
    assert score_map["c3"] == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-3)
