from types import SimpleNamespace

import numpy as np

from encapsulation.database.utils.sqlite_embedding_cache import SqliteEmbeddingCache
from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin


def test_sqlite_embedding_cache_roundtrip(tmp_path):
    db = SqliteEmbeddingCache(db_path=str(tmp_path / "cache.sqlite3"), max_in_keys_per_query=2)
    assert db.get_many(["k1"]) == {}

    v = np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    db.set_many([("k1", v, 3)])

    got = db.get_many(["k1", "missing"])
    assert "k1" in got
    blob, dim = got["k1"]
    assert dim == 3
    assert np.frombuffer(blob, dtype=np.float32, count=dim).tolist() == [1.0, 2.0, 3.0]


class _DummyEmbeddingModel:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(request_batch_size=64, embedding_dimensions=3)

    def embed(self, texts):
        # Deterministic 3d embedding based on simple hashing.
        self.calls.append(list(texts))
        out = []
        for t in texts:
            s = float(sum(ord(c) for c in t) % 1000)
            out.append([s, s + 1.0, s + 2.0])
        return out


class _DummyEmbedder(_PrunedHippoRAGNeo4jEmbeddingsMixin):
    def __init__(self, *, storage_path: str, cache_enabled: bool):
        self.storage_path = storage_path
        self.embedding_model = _DummyEmbeddingModel()
        self.config = SimpleNamespace(
            embedding=SimpleNamespace(
                type="openai_embedding",
                loading_method="openai",
                model_name="text-embedding-3-small",
                embedding_dimensions=3,
            ),
            embedding_cache=SimpleNamespace(
                enabled=bool(cache_enabled),
                dirname="embedding_cache",
                db_filename="embeddings.sqlite3",
                max_in_keys_per_query=500,
            ),
        )


def test_embed_dedup_and_owner_scoped_cache(tmp_path):
    d = _DummyEmbedder(storage_path=str(tmp_path), cache_enabled=True)

    texts = ["hello", "hello", "world"]
    out1 = d._embed_texts_resilient_scoped(texts, purpose="chunk", owner_id="ownerA")
    assert len(out1) == 3
    # Within one call, duplicates should be deduped -> only one embed() call for two unique texts.
    assert len(d.embedding_model.calls) == 1
    assert d.embedding_model.calls[0] == ["hello", "world"]

    # Second call with same owner + same texts should hit cache (no new embed calls).
    out2 = d._embed_texts_resilient_scoped(texts, purpose="chunk", owner_id="ownerA")
    assert out2 == out1
    assert len(d.embedding_model.calls) == 1

    # Different owner must not share cache: should cause another embed() call.
    _ = d._embed_texts_resilient_scoped(texts, purpose="chunk", owner_id="ownerB")
    assert len(d.embedding_model.calls) == 2

