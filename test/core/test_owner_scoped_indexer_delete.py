from dataclasses import dataclass

from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig


@dataclass
class _FakeFaissDB:
    deleted: list[list[str]]
    saved: list[str]

    # Mimic FaissVectorDB surface used by FaissIndexer.delete_chunks
    def delete_index(self, ids):  # noqa: ANN001
        self.deleted.append(list(ids or []))
        return True

    def save_index(self, index_dir: str):  # noqa: ANN001
        self.saved.append(str(index_dir))


@dataclass
class _FakeBM25Builder:
    deleted: list[list[str]]

    # Mimic BM25IndexBuilder surface used by BM25Indexer.delete_chunks
    def load_local(self) -> None:
        return None

    def delete_index(self, ids):  # noqa: ANN001
        self.deleted.append(list(ids or []))
        return True


def test_owner_scoped_faiss_delete_requires_owner_id() -> None:
    cfg = FaissIndexerConfig(
        index_config=FaissVectorDBConfig(
            embedding_config=QwenEmbeddingConfig(),
        ),
    )
    indexer = cfg.build()
    assert indexer._owner_scoped_enabled is True  # noqa: SLF001
    assert indexer.delete_chunks(["c1"]) is False


def test_owner_scoped_faiss_delete_uses_owner_db() -> None:
    cfg = FaissIndexerConfig(
        index_config=FaissVectorDBConfig(
            embedding_config=QwenEmbeddingConfig(),
        ),
    )
    indexer = cfg.build()
    fake = _FakeFaissDB(deleted=[], saved=[])
    indexer._db_by_owner["owner-1"] = fake  # noqa: SLF001

    ok = indexer.delete_chunks(["c1", "c2"], owner_id="owner-1")
    assert ok is True
    assert fake.deleted == [["c1", "c2"]]
    assert len(fake.saved) == 1


def test_owner_scoped_bm25_delete_requires_owner_id() -> None:
    cfg = BM25IndexerConfig(
        index_config=BM25BuilderConfig(),
    )
    indexer = cfg.build()
    assert indexer._owner_scoped_enabled is True  # noqa: SLF001
    assert indexer.delete_chunks(["c1"]) is False


def test_owner_scoped_bm25_delete_uses_owner_builder() -> None:
    cfg = BM25IndexerConfig(
        index_config=BM25BuilderConfig(),
    )
    indexer = cfg.build()
    fake = _FakeBM25Builder(deleted=[])
    indexer._builder_by_owner["owner-2"] = fake  # noqa: SLF001

    ok = indexer.delete_chunks(["c9"], owner_id="owner-2")
    assert ok is True
    assert fake.deleted == [["c9"]]

