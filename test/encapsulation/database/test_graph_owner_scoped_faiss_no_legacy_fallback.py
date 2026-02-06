import os
import pytest


def test_owner_scoped_faiss_db_does_not_fallback_to_shared_for_concrete_owner(monkeypatch, tmp_path):
    from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin

    class _Dummy(_PrunedHippoRAGNeo4jEmbeddingsMixin):
        def __init__(self) -> None:
            self.storage_path = str(tmp_path)
            self.fact_faiss_db = object()
            self.entity_faiss_db = object()

        def _faiss_owner_scoped_dir(self, kind: str, *, owner_id):  # noqa: ANN001
            base = os.path.join(self.storage_path, f"{kind}_index")
            leaf = "__GLOBAL__" if owner_id is None else str(owner_id)
            return os.path.join(base, leaf)

        @staticmethod
        def _faiss_index_artifacts_ready(_index_dir: str) -> bool:
            # Pretend only the legacy shared dir has artifacts.
            return _index_dir.endswith("/fact_index") or _index_dir.endswith("/entity_index")

        @staticmethod
        def _clone_faiss_config_for_path(_config, *, index_path: str):  # noqa: ANN001
            raise AssertionError("should not reach config clone for owner_id=None legacy fallback case")

    store = _Dummy()

    # Global scope may fall back to shared DB (legacy).
    assert store._owner_scoped_faiss_db("fact", owner_id=None) is store.fact_faiss_db

    # Concrete owner must NOT fall back to shared; it should proceed (and here we assert it does).
    with pytest.raises(AssertionError):
        store._owner_scoped_faiss_db("fact", owner_id="owner_1")

