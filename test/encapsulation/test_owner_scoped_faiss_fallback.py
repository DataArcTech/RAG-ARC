from pathlib import Path

from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin


class _TemplateFaissDb:
    def __init__(self, *, index_path: str) -> None:
        class _Cfg:
            def __init__(self, index_path: str) -> None:
                self.index_path = index_path

        self.config = _Cfg(index_path=index_path)


class _Dummy(_PrunedHippoRAGNeo4jEmbeddingsMixin):
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self, *, storage_path: Path) -> None:
        self.storage_path = str(storage_path)
        self.fact_faiss_db = _TemplateFaissDb(index_path=str(storage_path / "fact_index"))
        self.entity_faiss_db = _TemplateFaissDb(index_path=str(storage_path / "entity_index"))

    @classmethod
    def _restore_owner_id(cls, owner_id: str | None) -> str | None:
        if not owner_id or owner_id == cls.OWNER_GLOBAL_KEY:
            return None
        return owner_id


def test_owner_scoped_faiss_db_falls_back_to_legacy_layout(tmp_path: Path) -> None:
    storage = tmp_path / "graph_index"
    base_dir = storage / "fact_index"
    base_dir.mkdir(parents=True)
    (base_dir / "index.faiss").write_bytes(b"")
    (base_dir / "index.pkl").write_bytes(b"")

    store = _Dummy(storage_path=storage)
    db = store.get_fact_faiss_db("2a16b821-0e49-44c7-a5bb-96fd141f7772")
    assert db is store.fact_faiss_db

