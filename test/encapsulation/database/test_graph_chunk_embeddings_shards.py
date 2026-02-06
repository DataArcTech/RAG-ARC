import contextlib
from pathlib import Path

import numpy as np

from encapsulation.database.graph_db.pruned_hipporag_neo4j_chunk_embeddings import _PrunedHippoRAGNeo4jChunkEmbeddingsMixin
from encapsulation.database.graph_db.pruned_hipporag_neo4j_persistence import _PrunedHippoRAGNeo4jPersistenceMixin


class _Cfg:
    chunk_embeddings_owner_sharded = True
    chunk_embeddings_dirname = "chunk_embeddings"


class DummyStore(_PrunedHippoRAGNeo4jChunkEmbeddingsMixin, _PrunedHippoRAGNeo4jPersistenceMixin):
    def __init__(self, base: Path):
        self.config = _Cfg()
        self.storage_path = str(base)
        self.index_name = "index"
        self.chunk_embeddings = {}
        self._chunk_embeddings_array = None
        self._chunk_embedding_owner_by_chunk_id = {}
        self._chunk_ids_by_owner = {}
        self._chunk_embeddings_dirty_owners = set()

    def iter_owner_scoped_faiss_dbs(self, kind: str):
        return []  # not needed for these tests

    @contextlib.contextmanager
    def write_lock(self):
        yield

    @contextlib.contextmanager
    def read_lock(self):
        yield

    def _load_graph_cache(self, *args, **kwargs):
        return None

    def get_cache_version(self):
        return 0


def test_chunk_embeddings_sharded_save_and_load(tmp_path: Path):
    store = DummyStore(tmp_path)
    store.chunk_embeddings = {
        "c1": np.ones(3, dtype=np.float32),
        "c2": np.zeros(3, dtype=np.float32),
    }
    store._chunk_embedding_owner_by_chunk_id = {"c1": "o1", "c2": "o2"}
    store._chunk_ids_by_owner = {"o1": {"c1"}, "o2": {"c2"}}
    store._chunk_embeddings_dirty_owners = {"o1", "o2"}

    store.save_index(str(tmp_path), "index")

    shard_dir = tmp_path / "chunk_embeddings"
    assert shard_dir.is_dir()
    assert (shard_dir / "index_chunk_embeddings__o1.pkl").exists()
    assert (shard_dir / "index_chunk_embeddings__o2.pkl").exists()
    assert not (tmp_path / "index_chunk_embeddings.pkl").exists()

    store2 = DummyStore(tmp_path)
    store2._load_chunk_embeddings()
    assert set(store2.chunk_embeddings.keys()) == {"c1", "c2"}


def test_chunk_embeddings_save_only_dirty_owner(tmp_path: Path):
    store = DummyStore(tmp_path)
    store.chunk_embeddings = {"c1": np.ones(3, dtype=np.float32)}
    store._chunk_embedding_owner_by_chunk_id = {"c1": "o1"}
    store._chunk_ids_by_owner = {"o1": {"c1"}}
    store._chunk_embeddings_dirty_owners = {"o1"}
    store.save_index(str(tmp_path), "index")
    assert (tmp_path / "chunk_embeddings" / "index_chunk_embeddings__o1.pkl").exists()

    # Now mark no dirty changes; save should be no-op w.r.t. new files.
    store._chunk_embeddings_dirty_owners = set()
    store.save_index(str(tmp_path), "index")
    assert (tmp_path / "chunk_embeddings" / "index_chunk_embeddings__o1.pkl").exists()
