from pathlib import Path

from encapsulation.data_model.schema import Chunk
from encapsulation.database.vector_db.faiss import FaissVectorDB


class _DimlessEmbeddingConfig:
    def __init__(self, *, model_name: str):
        self.type = "stub_embedding"
        self.loading_method = "stub"
        self.model_name = model_name
        self.embedding_dimensions = None

    def build(self):
        return _DimlessEmbeddingLLM(self)


class _DimlessEmbeddingLLM:
    def __init__(self, config: _DimlessEmbeddingConfig):
        self.config = config

    def embed(self, texts):
        if isinstance(texts, str):
            return [0.01] * 8
        return [[0.01] * 8 for _ in list(texts or [])]


class _DimlessEmbeddingLLMRaises(_DimlessEmbeddingLLM):
    def embed(self, texts):
        raise RuntimeError("embed() should not be called during load_index()")


class _StubFaissConfig:
    def __init__(self, *, embedding_config: _DimlessEmbeddingConfig, index_type: str = "flat", metric: str = "cosine"):
        self.embedding_config = embedding_config
        self.index_type = index_type
        self.metric = metric
        self.normalize_L2 = True


def test_faiss_load_does_not_probe_embedding_dim(tmp_path: Path) -> None:
    index_dir = tmp_path / "faiss"

    cfg = _StubFaissConfig(embedding_config=_DimlessEmbeddingConfig(model_name="model-a"))
    db = FaissVectorDB(cfg)
    db.build_index([Chunk(id="c1", content="hello", owner_id="owner-1", metadata={"owner_id": "owner-1"})])
    db.save_index(str(index_dir))

    cfg2 = _StubFaissConfig(embedding_config=_DimlessEmbeddingConfig(model_name="model-a"))
    db2 = FaissVectorDB(cfg2)
    db2.embedding_model = _DimlessEmbeddingLLMRaises(cfg2.embedding_config)
    db2.load_index(str(index_dir))

