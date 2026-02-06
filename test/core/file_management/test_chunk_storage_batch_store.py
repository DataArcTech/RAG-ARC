import uuid

import pytest

from core.file_management.storage.chunk import ChunkStorage


class _StubBlobStore:
    def __init__(self) -> None:
        self.stored: dict[str, bytes] = {}

    def store(self, key: str, data: bytes, content_type=None, **kwargs):  # noqa: ANN001, ANN003
        self.stored[key] = data
        return key, False

    def exists(self, key: str, **kwargs):  # noqa: ANN001, ANN003
        return key in self.stored

    def delete(self, key: str, **kwargs):  # noqa: ANN001, ANN003
        return self.stored.pop(key, None) is not None


class _StubMetadataStore:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0
        self.parsed_calls = 0
        self.file_calls = 0
        self.deleted: list[str] = []

    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs):  # noqa: ANN001, ANN003
        self.parsed_calls += 1

        class _Meta:
            source_file_id = "f1"

        return _Meta()

    def get_file_metadata(self, file_id: str, **kwargs):  # noqa: ANN001, ANN003
        self.file_calls += 1

        class _Meta:
            owner_id = uuid.UUID(int=0)

        return _Meta()

    def store_chunk_metadata_batch(self, metas, **kwargs):  # noqa: ANN001, ANN003
        self.batch_calls += 1
        return [m.chunk_id for m in metas]

    def store_chunk_metadata(self, meta, **kwargs):  # noqa: ANN001, ANN003
        self.single_calls += 1
        return meta.chunk_id

    def update_chunk_metadata(self, chunk_id: str, updates, **kwargs):  # noqa: ANN001, ANN003
        return True

    def delete_chunk_metadata(self, chunk_id: str, **kwargs):  # noqa: ANN001, ANN003
        self.deleted.append(chunk_id)
        return True


class _Cfg:
    def __init__(self, blob, meta) -> None:
        class _B:
            def __init__(self, x):
                self._x = x

            def build(self):
                return self._x

        self.file_db_config = _B(blob)
        self.relational_db_config = _B(meta)


def test_chunk_storage_store_chunks_batch_uses_metadata_batch_insert():
    blob = _StubBlobStore()
    meta = _StubMetadataStore()
    storage = ChunkStorage(_Cfg(blob, meta))

    chunks = [{"content": "a", "metadata": {}}, {"content": "b", "metadata": {}}]
    refs = storage.store_chunks_batch(
        source_parsed_content_id="pc1",
        chunker_type="stub",
        chunks=chunks,
        owner_id=uuid.UUID(int=1),  # avoid per-chunk file metadata lookup
        validate_after_store=True,
    )

    assert len(refs) == 2
    assert meta.batch_calls == 1
    assert meta.single_calls == 0
    assert meta.parsed_calls == 1
    # owner_id provided -> no need to fetch file metadata
    assert meta.file_calls == 0


def test_chunk_storage_store_chunks_batch_returns_chunk_indices():
    blob = _StubBlobStore()
    meta = _StubMetadataStore()
    storage = ChunkStorage(_Cfg(blob, meta))

    chunks = [{"content": "a", "metadata": {}}, None, {"content": "c", "metadata": {}}]
    refs = storage.store_chunks_batch(
        source_parsed_content_id="pc1",
        chunker_type="stub",
        chunks=chunks,
        owner_id=uuid.UUID(int=1),
    )
    # chunk_index should preserve original indices (0 and 2).
    assert [i for i, _ in refs] == [0, 2]

