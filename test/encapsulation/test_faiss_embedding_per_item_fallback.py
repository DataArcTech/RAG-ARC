import shutil

import numpy as np
import pytest

from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from encapsulation.data_model.schema import Chunk
from encapsulation.database.vector_db.faiss import FaissVectorDB


class _EmbeddingStub:
    def embed(self, texts):  # noqa: ANN001
        # Simulate a flaky gateway: batch embedding fails, but per-item succeeds.
        if isinstance(texts, list):
            raise RuntimeError("timeout")
        return [1.0, 0.0, 0.0]


def test_faiss_vector_db_falls_back_to_per_item_embedding_on_batch_failure(tmp_path) -> None:
    FaissVectorDB.clear_cache()
    index_path = tmp_path / "faiss_embed_fallback"
    # Ensure a clean directory for the shared-module instance.
    if index_path.exists():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    cfg = FaissVectorDBConfig(
        index_path=str(index_path),
        index_type="flat",
        metric="cosine",
        normalize_L2=True,
        # Provide a valid config object, but override embedding_model in the instance
        # so the test never performs real network calls.
        embedding_config=OpenAIEmbeddingConfig(openai_api_key="test", openai_base_url="http://example.com"),
    )

    db = FaissVectorDB(cfg)
    db.embedding_model = _EmbeddingStub()  # type: ignore[assignment]

    chunks = [
        Chunk(id="c1", content="hello", metadata={}),
        Chunk(id="c2", content="world", metadata={}),
    ]
    out_ids = db.update_index(chunks) or []

    assert out_ids == ["c1", "c2"]
    assert db.index is not None
    assert int(db.index.ntotal) == 2

    # Ensure vectors are stored and normalized (cosine).
    v = np.zeros((1, 3), dtype=np.float32)
    db.index.reconstruct(0, v[0])
    assert pytest.approx(float(np.linalg.norm(v[0])), rel=1e-3, abs=1e-3) == 1.0


class _EmbeddingSplitStub:
    def __init__(self):
        self.calls: list[int] = []

    def embed(self, texts):  # noqa: ANN001
        if isinstance(texts, list):
            self.calls.append(len(texts))
            # Fail only for larger batches; smaller batches succeed.
            if len(texts) > 2:
                raise RuntimeError("batch_too_large")
            return [[1.0, 0.0, 0.0] for _ in texts]
        self.calls.append(1)
        return [1.0, 0.0, 0.0]


def test_faiss_vector_db_progressive_split_fallback_reduces_per_item_calls(tmp_path) -> None:
    FaissVectorDB.clear_cache()
    index_path = tmp_path / "faiss_embed_split_fallback"
    if index_path.exists():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    cfg = FaissVectorDBConfig(
        index_path=str(index_path),
        index_type="flat",
        metric="cosine",
        normalize_L2=True,
        embedding_config=OpenAIEmbeddingConfig(openai_api_key="test", openai_base_url="http://example.com"),
    )
    db = FaissVectorDB(cfg)
    stub = _EmbeddingSplitStub()
    db.embedding_model = stub  # type: ignore[assignment]

    chunks = [Chunk(id=f"c{i}", content=f"t{i}", metadata={}) for i in range(8)]
    out_ids = db.update_index(chunks) or []
    assert out_ids == [f"c{i}" for i in range(8)]

    # Should have attempted a failing big batch, then split into smaller ones (len=2).
    assert any(n > 2 for n in stub.calls)
    assert 2 in stub.calls
