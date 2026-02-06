import uuid
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from core.file_management.index_manager import IndexManager


@dataclass
class _Meta:
    file_id: str
    filename: str
    owner_id: uuid.UUID
    status: Any = None


class _FileStorage:
    def __init__(self) -> None:
        self.metadata_store = self
        self._meta = _Meta(file_id="f1", filename="a.txt", owner_id=uuid.UUID(int=0))

    def get_file_content(self, file_id: str):  # noqa: ARG002
        return b"hello world"

    def get_file_metadata(self, file_id: str):  # noqa: ARG002
        return self._meta

    def update_file_metadata(self, file_id: str, patch: Dict[str, Any], **kwargs):  # noqa: ARG002
        if "status" in patch:
            self._meta.status = patch["status"]
        return True

    def update_file_status(self, file_id: str, status, **kwargs):  # noqa: ANN001, ARG002
        self._meta.status = status
        return True


class _ParsedStorage:
    def __init__(self) -> None:
        self.metadata_store = self

    def store_parsed_content(self, **kwargs):  # noqa: ANN003
        return "pc1"

    def update_parsed_content_metadata(self, parsed_content_id: str, patch: Dict[str, Any], **kwargs):  # noqa: ARG002
        return True


class _ChunkStorage:
    def __init__(self) -> None:
        self.metadata_store = self
        self.batch_called = 0

    def store_chunks_batch(self, *, chunks, **kwargs):  # noqa: ANN001, ANN003
        self.batch_called += 1
        # Return (chunk_index, chunk_id) for every chunk.
        return [(i, f"c{i}") for i, _ in enumerate(chunks)]

    def store_chunk(self, **kwargs):  # noqa: ANN003
        raise AssertionError("store_chunk should not be called when store_chunks_batch exists")

    def update_chunk_metadata(self, chunk_id: str, patch: Dict[str, Any], **kwargs):  # noqa: ARG002
        return True


class _Parser:
    async def parse_file(self, *, file_data: bytes, filename: str, **kwargs):  # noqa: ARG002
        return {"content": file_data.decode("utf-8")}


class _Chunker:
    def get_chunker_info(self) -> Dict[str, Any]:
        return {"strategy": "stub"}

    def chunk_text(self, *, text: str, metadata: Dict[str, Any], **kwargs):  # noqa: ARG002
        return [{"content": text, "metadata": metadata} for _ in range(3)]


class _Indexer:
    async def update_index(self, chunk_objects):  # noqa: ANN001
        return [c.id for c in chunk_objects]


class _Cfg:
    def __init__(self) -> None:
        class _B:
            def __init__(self, x):
                self._x = x

            def build(self):
                return self._x

        self.file_storage_config = _B(_FileStorage())
        self.parsed_content_storage_config = _B(_ParsedStorage())
        self.chunk_storage_config = _B(_ChunkStorage())
        self.parser_config = _B(_Parser())
        self.chunker_config = _B(_Chunker())
        self.indexer_configs = [_B(_Indexer())]


@pytest.mark.asyncio
async def test_index_manager_prefers_chunk_batch_store_when_available():
    mgr = IndexManager(_Cfg())
    result = await mgr.index_file("f1")
    assert result.get("success") is True
    assert mgr.chunk_storage.batch_called == 1

