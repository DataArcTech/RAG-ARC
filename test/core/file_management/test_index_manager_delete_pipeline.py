from dataclasses import dataclass

from core.file_management.index_manager import IndexManager


@dataclass
class _ParsedContent:
    parsed_content_id: str
    source_file_id: str
    blob_key: str


@dataclass
class _ChunkMetadata:
    chunk_id: str
    source_parsed_content_id: str
    blob_key: str


class _BlobStore:
    def __init__(self):
        self.deleted: list[str] = []

    def delete(self, key: str, **kwargs):  # noqa: ARG002 - kwargs kept for parity
        self.deleted.append(key)


class _ParsedContentMetadataStore:
    def __init__(self, items: list[_ParsedContent]):
        self._items = {item.parsed_content_id: item for item in items}
        self.deleted_ids: list[str] = []

    def list_parsed_content_metadata(self, source_file_id: str, **kwargs):  # noqa: ARG002
        return [item for item in self._items.values() if item.source_file_id == source_file_id]

    def delete_parsed_content_metadata(self, parsed_content_id: str, **kwargs):  # noqa: ARG002
        self.deleted_ids.append(parsed_content_id)
        self._items.pop(parsed_content_id, None)


class _ChunkMetadataStore:
    def __init__(self, chunks: list[_ChunkMetadata]):
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.deleted_ids: list[str] = []

    def list_chunk_metadata(self, source_parsed_content_id: str, **kwargs):  # noqa: ARG002
        return [chunk for chunk in self._chunks.values() if chunk.source_parsed_content_id == source_parsed_content_id]

    def get_chunk_metadata(self, chunk_id: str, **kwargs):  # noqa: ARG002
        return self._chunks.get(chunk_id)

    def delete_chunk_metadata(self, chunk_id: str, **kwargs):  # noqa: ARG002
        self.deleted_ids.append(chunk_id)
        self._chunks.pop(chunk_id, None)


class _StubParsedContentStorage:
    def __init__(self, items: list[_ParsedContent]):
        self.metadata_store = _ParsedContentMetadataStore(items)
        self.blob_store = _BlobStore()


class _StubChunkStorage:
    def __init__(self, chunks: list[_ChunkMetadata]):
        self.metadata_store = _ChunkMetadataStore(chunks)
        self.blob_store = _BlobStore()


class _StubIndexer:
    def __init__(self, success: bool):
        self.success = success
        self.calls: list[list[str]] = []

    def delete_chunks(self, chunk_ids: list[str]) -> bool:
        self.calls.append(list(chunk_ids))
        return self.success


def _make_manager(*, indexer_success: bool):
    parsed_items = [_ParsedContent(parsed_content_id="pc-1", source_file_id="file-1", blob_key="pc-blob")]
    chunk_items = [_ChunkMetadata(chunk_id="chunk-1", source_parsed_content_id="pc-1", blob_key="chunk-blob")]

    parsed_storage = _StubParsedContentStorage(parsed_items)
    chunk_storage = _StubChunkStorage(chunk_items)
    indexer = _StubIndexer(success=indexer_success)

    manager = object.__new__(IndexManager)
    manager.parsed_content_storage = parsed_storage
    manager.chunk_storage = chunk_storage
    manager.indexers = [indexer]
    manager.file_storage = None  # Unused in delete pipeline
    manager.parser = None
    manager.chunker = None

    return manager, parsed_storage, chunk_storage


def test_delete_file_data_aborts_when_indexer_fails():
    manager, parsed_storage, chunk_storage = _make_manager(indexer_success=False)

    result = manager.delete_file_data("file-1")

    assert result["success"] is False
    assert result["error_message"] == "Failed to delete chunks from some indexers"
    assert chunk_storage.blob_store.deleted == []
    assert chunk_storage.metadata_store.deleted_ids == []
    assert parsed_storage.blob_store.deleted == []
    assert parsed_storage.metadata_store.deleted_ids == []
    assert chunk_storage.metadata_store.get_chunk_metadata("chunk-1") is not None


def test_delete_file_data_removes_metadata_after_success():
    manager, parsed_storage, chunk_storage = _make_manager(indexer_success=True)

    result = manager.delete_file_data("file-1")

    assert result["success"] is True
    assert chunk_storage.blob_store.deleted == ["chunk-blob"]
    assert chunk_storage.metadata_store.deleted_ids == ["chunk-1"]
    assert parsed_storage.blob_store.deleted == ["pc-blob"]
    assert parsed_storage.metadata_store.deleted_ids == ["pc-1"]

