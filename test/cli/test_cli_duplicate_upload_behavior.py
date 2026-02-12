import uuid
from pathlib import Path

import cli.rag as rag
from core.file_management.storage.file import FileValidationError


class _StubFileStorage:
    def __init__(self, *, duplicate_ids: list[str], duplicate_on_first_upload: bool, uploaded_file_id: str = "new-file-id"):
        self.duplicate_ids = duplicate_ids
        self.duplicate_on_first_upload = duplicate_on_first_upload
        self.uploaded_file_id = uploaded_file_id
        self._duplicate_raised = False
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
        if self.duplicate_on_first_upload and not self._duplicate_raised:
            self._duplicate_raised = True
            raise FileValidationError(f"File with name {filename} already exists")
        return self.uploaded_file_id


class _StubFileIndex:
    def __init__(self):
        self.index_calls: list[tuple[str, bool]] = []
        self.parse_calls: list[tuple[str, bool]] = []

    async def index_file(self, file_id: str, force_reparse: bool = False):
        self.index_calls.append((file_id, force_reparse))
        return {"success": True, "chunk_ids": ["c1"]}

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


def _make_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "demo.pdf"
    path.write_bytes(b"pdf-bytes")
    return path


def test_ingest_duplicate_reuses_existing_when_not_force_reparse(tmp_path: Path):
    path = _make_pdf(tmp_path)
    owner_id = uuid.uuid4()
    storage = _StubFileStorage(duplicate_ids=["existing-1"], duplicate_on_first_upload=True)
    knowledge = _StubKnowledge(storage)

    ok = rag._ingest_single_file(
        path,
        knowledge,
        owner_id,
        logical_filename="RAG-ARC/demo.pdf",
        force_reparse=False,
    )

    assert ok is True
    assert len(storage.upload_calls) == 1
    assert len(storage.find_duplicate_calls) == 1
    assert knowledge.deleted_calls == []
    assert knowledge.file_index.index_calls == [("existing-1", False)]


def test_ingest_duplicate_force_reparse_deletes_and_reuploads(tmp_path: Path):
    path = _make_pdf(tmp_path)
    owner_id = uuid.uuid4()
    storage = _StubFileStorage(duplicate_ids=["old-1", "old-2"], duplicate_on_first_upload=True)
    knowledge = _StubKnowledge(storage)

    ok = rag._ingest_single_file(
        path,
        knowledge,
        owner_id,
        logical_filename="RAG-ARC/demo.pdf",
        force_reparse=True,
    )

    assert ok is True
    assert len(storage.upload_calls) == 2
    assert len(storage.find_duplicate_calls) == 1
    assert [file_id for file_id, _ in knowledge.deleted_calls] == ["old-1", "old-2"]
    assert knowledge.file_index.index_calls == [("new-file-id", True)]


def test_parse_duplicate_reuses_existing_when_not_force_reparse(tmp_path: Path):
    path = _make_pdf(tmp_path)
    owner_id = uuid.uuid4()
    storage = _StubFileStorage(duplicate_ids=["existing-p"], duplicate_on_first_upload=True)
    knowledge = _StubKnowledge(storage)

    ok = rag._parse_single_file(
        path,
        knowledge,
        owner_id,
        logical_filename="RAG-ARC/demo.pdf",
        force_reparse=False,
    )

    assert ok is True
    assert len(storage.upload_calls) == 1
    assert len(storage.find_duplicate_calls) == 1
    assert knowledge.deleted_calls == []
    assert knowledge.file_index.parse_calls == [("existing-p", False)]


def test_parse_duplicate_force_reparse_deletes_and_reuploads(tmp_path: Path):
    path = _make_pdf(tmp_path)
    owner_id = uuid.uuid4()
    storage = _StubFileStorage(duplicate_ids=["old-1"], duplicate_on_first_upload=True)
    knowledge = _StubKnowledge(storage)

    ok = rag._parse_single_file(
        path,
        knowledge,
        owner_id,
        logical_filename="RAG-ARC/demo.pdf",
        force_reparse=True,
    )

    assert ok is True
    assert len(storage.upload_calls) == 2
    assert len(storage.find_duplicate_calls) == 1
    assert [file_id for file_id, _ in knowledge.deleted_calls] == ["old-1"]
    assert knowledge.file_index.parse_calls == [("new-file-id", True)]
