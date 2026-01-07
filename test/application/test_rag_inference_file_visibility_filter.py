import uuid

import pytest

from application.rag_inference.module import RAGInference
from encapsulation.data_model.schema import Chunk


class _StubKnowledge:
    def __init__(self, inactive_file_ids: set[str]):
        self._inactive_file_ids = inactive_file_ids

    def is_file_active(self, file_id: str) -> bool:
        return file_id not in self._inactive_file_ids


@pytest.mark.parametrize(
    ("metadata", "expected_kept"),
    [
        ({"source_file_id": "file-deleted"}, False),
        ({"sourceFileId": "file-deleted"}, False),
        ({"file_id": "file-deleted"}, False),
        ({"document_id": "file-deleted"}, False),
        ({"doc_id": "file-deleted"}, False),
        ({"chunk_metadata": {"source_file_id": "file-deleted"}}, False),
        ({"chunk_metadata": {"sourceFileId": "file-deleted"}}, False),
        ({}, True),
        (None, True),
    ],
)
def test_rag_inference_filters_chunks_from_deleted_files(metadata, expected_kept):
    rag = RAGInference.__new__(RAGInference)
    rag._knowledge_module = _StubKnowledge(inactive_file_ids={"file-deleted"})

    chunk = Chunk(
        id="chunk-1",
        content="hello",
        owner_id=str(uuid.uuid4()),
        metadata=metadata,
        graph=None,
    )
    result = rag._filter_chunks_by_file_status([chunk])

    assert (len(result) == 1) is expected_kept

