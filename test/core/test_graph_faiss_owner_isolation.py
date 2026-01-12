import pickle
from pathlib import Path

import numpy as np

from core.utils.rwlock import RWLock
from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin
from encapsulation.database.vector_db.faiss import FaissVectorDB


class _StubEmbeddingConfig:
    def __init__(self, *, dim: int = 8):
        self.type = "stub_embedding"
        self.loading_method = "stub"
        self.model_name = "stub"
        self.embedding_dimensions = int(dim)

    def build(self):
        return _StubEmbeddingLLM(self)


class _StubEmbeddingLLM:
    def __init__(self, config: _StubEmbeddingConfig):
        self.config = config

    def embed(self, texts):  # noqa: ANN001
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in list(texts or []):
            seed = abs(hash(t)) % 997
            vec = np.linspace(0.0, 1.0, self.config.embedding_dimensions, dtype=np.float32) + float(seed) * 1e-4
            out.append(vec.tolist())
        return out


class _StubFaissConfig:
    def __init__(self, *, embedding_config: _StubEmbeddingConfig, index_type: str, index_path: str):
        self.embedding_config = embedding_config
        self.index_type = index_type
        self.index_path = str(index_path)
        self.metric = "cosine"
        self.normalize_L2 = True
        self.nlist = 32
        self.m = 8
        self.efConstruction = 40
        self.efSearch = 16
        self.train_size = 10000


class _FakeStore(_PrunedHippoRAGNeo4jEmbeddingsMixin):
    def __init__(self, storage_path: Path):
        self.storage_path = str(storage_path)
        self.index_name = "index"
        self.embedding_model = _StubEmbeddingLLM(_StubEmbeddingConfig(dim=8))
        self.chunk_embeddings = {}
        self._chunk_embeddings_array = None
        self._rwlock = RWLock()

        embed_cfg = _StubEmbeddingConfig(dim=8)
        self.entity_faiss_db = FaissVectorDB(
            _StubFaissConfig(embedding_config=embed_cfg, index_type="hnsw", index_path=str(storage_path / "_tmpl_entity"))
        )
        self.fact_faiss_db = FaissVectorDB(
            _StubFaissConfig(embedding_config=embed_cfg, index_type="flat", index_path=str(storage_path / "_tmpl_fact"))
        )

    def read_lock(self):
        return self._rwlock.read_lock()

    def write_lock(self):
        return self._rwlock.write_lock()

    @staticmethod
    def _restore_owner_id(owner_id):  # noqa: ANN001
        if not owner_id or owner_id == "__GLOBAL__":
            return None
        return str(owner_id)

    @staticmethod
    def _owner_key(owner_id):  # noqa: ANN001
        return str(owner_id) if owner_id else "__GLOBAL__"

    def _execute_query(self, query: str, params=None):  # noqa: ANN001
        if "MATCH (c:Chunk)" in query:
            return []
        if "MATCH (e:Entity)" in query and "RETURN e.entity_id AS entity_id" in query:
            return [
                {"entity_id": "entity-a1", "entity_name": "Alpha", "owner_id": "owner-a"},
                {"entity_id": "entity-a2", "entity_name": "Alpha2", "owner_id": "owner-a"},
                {"entity_id": "entity-b1", "entity_name": "Beta", "owner_id": "owner-b"},
            ]
        if "MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)" in query:
            return [
                {
                    "fact_id": "fact-a1",
                    "text": "Alpha | related_to | Alpha2",
                    "predicate": "related_to",
                    "owner_id": "owner-a",
                    "source_chunk_ids": ["chunk-a"],
                    "source_chunk_ids_truncated": False,
                    "head_id": "entity-a1",
                    "tail_id": "entity-a2",
                    "head_name": "Alpha",
                    "tail_name": "Alpha2",
                    "head_type": "Entity",
                    "tail_type": "Entity",
                },
                {
                    "fact_id": "fact-b1",
                    "text": "Beta | related_to | Beta",
                    "predicate": "related_to",
                    "owner_id": "owner-b",
                    "source_chunk_ids": ["chunk-b"],
                    "source_chunk_ids_truncated": False,
                    "head_id": "entity-b1",
                    "tail_id": "entity-b1",
                    "head_name": "Beta",
                    "tail_name": "Beta",
                    "head_type": "Entity",
                    "tail_type": "Entity",
                },
            ]
        raise AssertionError(f"Unexpected query: {query!r}")


def _owners_in_index(pkl_path: Path) -> set[str]:
    data = pickle.loads(pkl_path.read_bytes())
    docstore = data.get("docstore", {})
    owners: set[str] = set()
    for chunk in docstore.values():
        owner = getattr(chunk, "owner_id", None)
        if owner is None and getattr(chunk, "metadata", None):
            owner = chunk.metadata.get("owner_id")
        if owner is not None:
            owners.add(str(owner))
    return owners


def test_graph_fact_entity_faiss_are_owner_scoped_on_disk(tmp_path: Path) -> None:
    store = _FakeStore(tmp_path)
    store.batch_generate_embeddings()

    # Expect owner-scoped directories; a single shared index file causes cross-run owner mixing.
    fact_a = tmp_path / "fact_index" / "owner-a" / "index.pkl"
    fact_b = tmp_path / "fact_index" / "owner-b" / "index.pkl"
    entity_a = tmp_path / "entity_index" / "owner-a" / "index.pkl"
    entity_b = tmp_path / "entity_index" / "owner-b" / "index.pkl"

    assert fact_a.exists()
    assert fact_b.exists()
    assert entity_a.exists()
    assert entity_b.exists()

    assert _owners_in_index(fact_a) == {"owner-a"}
    assert _owners_in_index(fact_b) == {"owner-b"}
    assert _owners_in_index(entity_a) == {"owner-a"}
    assert _owners_in_index(entity_b) == {"owner-b"}
