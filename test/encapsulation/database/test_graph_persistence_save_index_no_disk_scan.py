def test_graph_persistence_save_index_does_not_scan_owner_dirs(monkeypatch, tmp_path):
    """save_index should not call iter_owner_scoped_faiss_dbs (which scans/loads every owner index)."""

    from encapsulation.database.graph_db.pruned_hipporag_neo4j_persistence import _PrunedHippoRAGNeo4jPersistenceMixin

    class _Dummy(_PrunedHippoRAGNeo4jPersistenceMixin):
        def __init__(self) -> None:
            self.chunk_embeddings = {}

            class _DB:
                def save_index(self, *_a, **_k):  # noqa: ANN001
                    return None

            self.fact_faiss_db = _DB()
            self.entity_faiss_db = _DB()

        def iter_owner_scoped_faiss_dbs(self, *_a, **_k):  # noqa: ANN001
            raise AssertionError("save_index must not scan disk owner dirs")

        def _save_chunk_embeddings(self, *, base_path: str, name: str):  # noqa: ANN001
            (tmp_path / "chunks.ok").write_text("ok", encoding="utf-8")

    store = _Dummy()
    store.save_index(str(tmp_path), "index")
    assert (tmp_path / "chunks.ok").exists()

