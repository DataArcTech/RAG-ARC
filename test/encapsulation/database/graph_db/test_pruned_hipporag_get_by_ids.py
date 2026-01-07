import json

from encapsulation.data_model.schema import GraphData
from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing import _PrunedHippoRAGNeo4jIndexingMixin


class _StubStore(_PrunedHippoRAGNeo4jIndexingMixin):
    def __init__(self, records_by_id: dict[str, dict]):
        self._records_by_id = records_by_id

    def _execute_query(self, query: str, params=None):  # noqa: ANN001
        chunk_id = (params or {}).get("chunk_id")
        if not chunk_id:
            return []
        record = self._records_by_id.get(chunk_id)
        return [record] if record else []

    def _get_graph_data(self, chunk_id: str):  # noqa: ARG002
        return GraphData(entities=[], relations=[], metadata={})


def test_get_by_ids_backfills_source_file_id_when_missing_in_metadata():
    store = _StubStore(
        {
            "chunk-1": {
                "chunk_id": "chunk-1",
                "content": "hello",
                "owner_id": "owner-1",
                "metadata": json.dumps({}),
                "source_file_id": "file-123",
            }
        }
    )

    chunks = store.get_by_ids(["chunk-1"])

    assert chunks[0].metadata["source_file_id"] == "file-123"

