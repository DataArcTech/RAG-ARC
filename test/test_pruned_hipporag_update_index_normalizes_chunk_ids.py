import uuid
from contextlib import contextmanager
from typing import Iterator, List

from encapsulation.data_model.schema import Chunk
from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing_ops import _PrunedHippoRAGNeo4jIndexingOpsMixin


class _DummyStore(_PrunedHippoRAGNeo4jIndexingOpsMixin):
    def __init__(self) -> None:
        self._cache_version = 0
        self.add_synonymy_edges = False
        self.captured_chunk_ids: List[str] | None = None
        self.captured_entity_ids: List[str] | None = None

    def _batch_add_chunks_and_graph_data(self, chunks: List[Chunk], **kwargs):  # noqa: ANN001
        # Simulate successful ingest.
        return [], {"entity_mentions": {"enabled": False, "attempted": 0, "written": 0, "elapsed_s": 0.0}}

    def batch_generate_embeddings(self, *, chunk_ids=None, entity_ids=None):  # noqa: ANN001
        # The bug: chunk_ids could be UUID objects and would not match Neo4j string ids.
        assert chunk_ids is not None
        assert all(isinstance(x, str) for x in chunk_ids)
        self.captured_chunk_ids = list(chunk_ids)
        self.captured_entity_ids = list(entity_ids or [])
        return {}

    def _update_graph_cache_incremental(self, new_chunk_ids, new_entity_ids):  # noqa: ANN001, ARG002
        return

    def _append_chunk_embeddings(self, new_chunk_ids):  # noqa: ANN001, ARG002
        return

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        yield None


def test_update_index_normalizes_chunk_ids_to_strings() -> None:
    store = _DummyStore()
    c1 = Chunk(id=uuid.UUID("00000000-0000-0000-0000-000000000001"), content="x", metadata={})
    c2 = Chunk(id="00000000-0000-0000-0000-000000000002", content="y", metadata={})
    ok = store.update_index([c1, c2])
    assert ok is True
    assert store.captured_chunk_ids == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
