from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing_ops import (
    _PrunedHippoRAGNeo4jIndexingOpsMixin,
)


class _NoopFaissDb:
    def delete_index(self, ids: List[str]) -> None:  # noqa: ARG002
        return

    def save_index(self, path: str, name: str) -> None:  # noqa: ARG002
        return


class _DummyStore(_PrunedHippoRAGNeo4jIndexingOpsMixin):
    def __init__(self, *, orphan_entities: Optional[List[str]] = None, orphan_fact_ids: Optional[List[str]] = None) -> None:
        self.chunk_embeddings: Dict[str, Any] = {}
        self._chunk_embeddings_array = None
        self._chunk_ids_list = None
        self._cache_version = 0

        self._stub_orphan_entities = list(orphan_entities or [])
        self._stub_orphan_fact_ids = list(orphan_fact_ids or [])
        self.invalidated: tuple[List[str], List[str]] | None = None

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None):  # noqa: ANN001
        params = params or {}
        if "MATCH (c:Chunk" in query and "WHERE size(all_chunks) = size(deleted_chunks)" in query:
            return [
                {"entity_id": entity_id, "entity_name": entity_id, "owner_id": "00000000-0000-0000-0000-000000000001"}
                for entity_id in self._stub_orphan_entities
            ]
        if "MATCH (e:Entity" in query and "RETURN DISTINCT r.fact_id AS fact_id" in query:
            return [
                {"fact_id": fact_id, "owner_id": "00000000-0000-0000-0000-000000000001"}
                for fact_id in self._stub_orphan_fact_ids
            ]
        return []

    def _restore_owner_id(self, owner_id: Any) -> str | None:
        if owner_id is None:
            return None
        return str(owner_id)

    def get_fact_faiss_db(self, owner: Any) -> _NoopFaissDb:  # noqa: ARG002
        return _NoopFaissDb()

    def get_entity_faiss_db(self, owner: Any) -> _NoopFaissDb:  # noqa: ARG002
        return _NoopFaissDb()

    def _faiss_owner_scoped_dir(self, kind: str, *, owner_id: Any | None = None) -> str:  # noqa: ARG002
        return "/tmp"

    def _invalidate_graph_cache_for_deleted_nodes(self, chunk_ids: List[str], entity_ids: List[str]) -> None:
        self.invalidated = (list(chunk_ids), list(entity_ids))

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        yield None


def test_delete_chunks_no_orphans_does_not_raise() -> None:
    store = _DummyStore()
    ok = store.delete_chunks(["c1", "c2"])
    assert ok is True
    assert store.invalidated == (["c1", "c2"], [])


def test_delete_chunks_with_orphans_propagates_orphan_ids_to_cache_invalidation() -> None:
    store = _DummyStore(orphan_entities=["e1", "e2"], orphan_fact_ids=["f1", "f2"])
    ok = store.delete_chunks(["c1"])
    assert ok is True
    assert store.invalidated == (["c1"], ["e1", "e2"])

