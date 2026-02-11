import uuid
from pathlib import Path

from core.file_management.storage.file import FileValidationError
import cli.rag as rag


class _StubFileStorage:
    def __init__(self, *, duplicate_ids: list[str], raise_duplicate_once: bool, uploaded_file_id: str = "new-file-id"):
        self.duplicate_ids = duplicate_ids
        self.raise_duplicate_once = raise_duplicate_once
        self.uploaded_file_id = uploaded_file_id
        self.upload_calls: list[dict] = []
        self.find_duplicate_calls: list[dict] = []

    def find_duplicate_file_ids(self, *, filename: str, file_data: bytes, owner_id: uuid.UUID) -> list[str]:
        self.find_duplicate_calls.append(
            {
                "filename": filename,
                "owner_id": owner_id,
                "size": len(file_data),
            }
        )
        return list(self.duplicate_ids)

    def upload_file(self, *, filename: str, file_data: bytes, owner_id: uuid.UUID, content_type: str) -> str:
        self.upload_calls.append(
            {
                "filename": filename,
                "owner_id": owner_id,
                "size": len(file_data),
                "content_type": content_type,
            }
        )
        if self.raise_duplicate_once:
            self.raise_duplicate_once = False
            raise FileValidationError(f"File with name {filename} already exists")
        return self.uploaded_file_id


class _StubFileIndex:
    def __init__(self):
        self.parse_calls: list[tuple[str, bool]] = []

    async def parse_file(self, file_id: str, force_reparse: bool = False):
        self.parse_calls.append((file_id, force_reparse))
        return {"success": True, "parsed_content_id": "pc-1"}


class _StubKnowledge:
    def __init__(self, file_storage: _StubFileStorage):
        self.file_storage = file_storage
        self.file_index = _StubFileIndex()
        self.deleted_calls: list[tuple[str, uuid.UUID]] = []

    async def mark_file_deleted_cli(self, file_id: str, owner_id: uuid.UUID):
        self.deleted_calls.append((file_id, owner_id))
        return {"status": "marked", "file_id": file_id}


def test_parse_single_file_force_reparse_deletes_duplicates_and_reuploads(tmp_path: Path):
    path = tmp_path / "demo.pdf"
    path.write_bytes(b"pdf-bytes")
    owner_id = uuid.uuid4()

    storage = _StubFileStorage(duplicate_ids=["old-1", "old-2"], raise_duplicate_once=True)
    knowledge = _StubKnowledge(storage)

    ok = rag._parse_single_file(
        path,
        knowledge,
        owner_id,
        logical_filename="RAG-ARC/demo.pdf",
        force_reparse=True,
    )

    assert ok is True
    assert [file_id for file_id, _ in knowledge.deleted_calls] == ["old-1", "old-2"]
    assert len(storage.find_duplicate_calls) == 1
    assert len(storage.upload_calls) == 2
    assert knowledge.file_index.parse_calls == [("new-file-id", True)]


def test_parse_single_file_without_force_reparse_keeps_duplicate_error(tmp_path: Path):
    path = tmp_path / "demo.pdf"
    path.write_bytes(b"pdf-bytes")
    owner_id = uuid.uuid4()

    storage = _StubFileStorage(duplicate_ids=["old-1"], raise_duplicate_once=True)
    knowledge = _StubKnowledge(storage)

    ok = rag._parse_single_file(
        path,
        knowledge,
        owner_id,
        logical_filename="RAG-ARC/demo.pdf",
        force_reparse=False,
    )

    assert ok is False
    assert knowledge.deleted_calls == []
    assert storage.find_duplicate_calls == []
    assert len(storage.upload_calls) == 1
    assert knowledge.file_index.parse_calls == []
