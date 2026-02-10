from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import uuid

from core.file_management.index_manager_deletion import _IndexManagerDeletionMixin


@dataclass
class _ParsedContentMeta:
    parsed_content_id: str
    blob_key: str
    owner_id: uuid.UUID


@dataclass
class _ChunkMeta:
    chunk_id: str
    blob_key: str


class _ListStore:
    def __init__(self, parsed: list[_ParsedContentMeta], chunks_by_parsed: dict[str, list[_ChunkMeta]]):
        self._parsed = parsed
        self._chunks_by_parsed = chunks_by_parsed
        self.deleted_parsed: list[str] = []
        self.deleted_chunks: list[str] = []

    # Parsed content metadata
    def list_parsed_content_metadata(self, *, source_file_id: str, **_kwargs: Any):
        return list(self._parsed)

    def delete_parsed_content_metadata(self, parsed_content_id: str, **_kwargs: Any) -> bool:
        self.deleted_parsed.append(parsed_content_id)
        return True

    # Chunk metadata
    def list_chunk_metadata(self, *, source_parsed_content_id: str, **_kwargs: Any):
        return list(self._chunks_by_parsed.get(source_parsed_content_id, []))

    def get_chunk_metadata(self, chunk_id: str, **_kwargs: Any):
        for rows in self._chunks_by_parsed.values():
            for row in rows:
                if row.chunk_id == chunk_id:
                    return row
        return None

    def delete_chunk_metadata(self, chunk_id: str, **_kwargs: Any) -> bool:
        self.deleted_chunks.append(chunk_id)
        return True


class _BlobStore:
    def __init__(self):
        self.deleted: list[str] = []

    def delete(self, blob_key: str, **_kwargs: Any) -> bool:
        self.deleted.append(str(blob_key))
        return True


class _FileStorageStub:
    def __init__(self, owner_id: uuid.UUID):
        self._owner_id = owner_id

    def get_file_metadata(self, file_id: str, **_kwargs: Any):
        return type("_Meta", (), {"owner_id": self._owner_id, "file_id": file_id})()


class _IndexStorage:
    def __init__(self, metadata_store: _ListStore, blob_store: _BlobStore):
        self.metadata_store = metadata_store
        self.blob_store = blob_store


class _IndexerStub:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def delete_chunks(self, chunk_ids: List[str], **kwargs: Any) -> bool:
        self.calls.append({"chunk_ids": list(chunk_ids), "kwargs": dict(kwargs)})
        return True


class _DeletionHarness(_IndexManagerDeletionMixin):
    """Minimal host object providing storages/indexers to exercise delete_file_data."""

    def __init__(self, *, owner_id: uuid.UUID, parsed: list[_ParsedContentMeta], chunks_by_parsed: dict[str, list[_ChunkMeta]]):
        self.file_storage = _FileStorageStub(owner_id)
        store = _ListStore(parsed=parsed, chunks_by_parsed=chunks_by_parsed)
        self.parsed_content_storage = _IndexStorage(metadata_store=store, blob_store=_BlobStore())
        self.chunk_storage = _IndexStorage(metadata_store=store, blob_store=_BlobStore())
        # 3 routes: graph + dense + bm25 (behavior is identical for deletion mixin; order should not matter).
        self.indexers = [_IndexerStub("graph"), _IndexerStub("faiss"), _IndexerStub("bm25")]


def test_delete_file_data_deletes_all_indexers_and_blobs_preserve_parsed_content() -> None:
    owner_id = uuid.uuid4()
    file_id = "file-123"
    parsed_id = "parsed-1"
    parsed = [_ParsedContentMeta(parsed_content_id=parsed_id, blob_key="parsed/blob", owner_id=owner_id)]
    chunks = [
        _ChunkMeta(chunk_id="chunk-1", blob_key="chunks/blob1"),
        _ChunkMeta(chunk_id="chunk-2", blob_key="chunks/blob2"),
    ]
    harness = _DeletionHarness(owner_id=owner_id, parsed=parsed, chunks_by_parsed={parsed_id: chunks})

    result = harness.delete_file_data(file_id, preserve_parsed_content=True)

    assert result["success"] is True
    # All indexers get the same chunk_ids and an owner_id context is present.
    for indexer in harness.indexers:
        assert len(indexer.calls) == 1
        assert indexer.calls[0]["chunk_ids"] == ["chunk-1", "chunk-2"]
        assert str(indexer.calls[0]["kwargs"].get("owner_id")) == str(owner_id)

    # Chunk blobs + chunk metadata are deleted.
    assert set(harness.chunk_storage.blob_store.deleted) == {"chunks/blob1", "chunks/blob2"}
    assert set(harness.chunk_storage.metadata_store.deleted_chunks) == {"chunk-1", "chunk-2"}

    # Parsed content is preserved (no parsed blob/metadata deletions).
    assert harness.parsed_content_storage.blob_store.deleted == []
    assert harness.parsed_content_storage.metadata_store.deleted_parsed == []

