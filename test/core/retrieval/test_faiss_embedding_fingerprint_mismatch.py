from pathlib import Path

import pytest

from encapsulation.data_model.schema import Chunk
from encapsulation.database.vector_db.faiss import FaissVectorDB


class _StubEmbeddingConfig:
    def __init__(self, *, model_name: str, embedding_dimensions: int):
        self.type = "stub_embedding"
        self.loading_method = "stub"
        self.model_name = model_name
        self.embedding_dimensions = int(embedding_dimensions)

    def build(self):
        return _StubEmbeddingLLM(self)


class _StubEmbeddingLLM:
    def __init__(self, config: _StubEmbeddingConfig):
        self.config = config

    def embed(self, texts):
        dim = int(self.config.embedding_dimensions)
        if isinstance(texts, str):
            return [0.01] * dim
        return [[0.01] * dim for _ in list(texts or [])]


class _StubFaissConfig:
    def __init__(self, *, embedding_config: _StubEmbeddingConfig, index_type: str = "flat", metric: str = "cosine"):
        self.embedding_config = embedding_config
        self.index_type = index_type
        self.metric = metric
        self.normalize_L2 = True


def test_faiss_load_rejects_embedding_fingerprint_mismatch(tmp_path: Path) -> None:
    index_dir = tmp_path / "faiss"

    cfg_a = _StubFaissConfig(embedding_config=_StubEmbeddingConfig(model_name="model-a", embedding_dimensions=8))
    db_a = FaissVectorDB(cfg_a)
    db_a.build_index([Chunk(id="c1", content="hello", owner_id="owner-1", metadata={"owner_id": "owner-1"})])
    db_a.save_index(str(index_dir))

    cfg_b = _StubFaissConfig(embedding_config=_StubEmbeddingConfig(model_name="model-b", embedding_dimensions=8))
    db_b = FaissVectorDB(cfg_b)
    with pytest.raises(ValueError, match="fingerprint"):
        db_b.load_index(str(index_dir))

