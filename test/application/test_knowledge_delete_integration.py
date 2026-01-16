import asyncio
import uuid
from dataclasses import dataclass, field

import pytest

from application.knowledge.module import Knowledge
from encapsulation.data_model.orm_models import FileStatus


@dataclass
class _StubFileMetadata:
    file_id: str
    owner_id: uuid.UUID
    status: FileStatus
    filename: str = "stub.txt"
    blob_key: str = "blob-key"
    file_size: int = 0
    content_type: str = "text/plain"
    created_at: float = field(default_factory=lambda: 0.0)
    updated_at: float = field(default_factory=lambda: 0.0)


class _StubFileStorage:
    def __init__(self, metadata: _StubFileMetadata):
        self._metadata = metadata
        self.metadata_store = self
        self.status_updates: list[FileStatus] = []
        self.deleted_files: list[str] = []

    # File metadata helpers -------------------------------------------------
    def get_file_metadata(self, file_id: str):
        return self._metadata if file_id == self._metadata.file_id else None

    def list_accessible_files(self, user_id, status=None, limit=None, offset=None, search=None):  # noqa: ANN001
        files = []
        if self._metadata.owner_id == user_id:
            if status is None or self._metadata.status == status:
                files.append(self._metadata)
        return files

    def count_accessible_files(self, user_id, status=None, search=None):  # noqa: ANN001
        return len(self.list_accessible_files(user_id=user_id, status=status, search=search))

    def update_file_status(self, file_id: str, status: FileStatus):
        if file_id != self._metadata.file_id:
            raise AssertionError("update_file_status called with unexpected ID")
        self._metadata.status = status
        self.status_updates.append(status)
        return True

    # Physical file deletion ------------------------------------------------
    def delete_file(self, file_id: str) -> bool:
        if file_id != self._metadata.file_id:
            raise AssertionError("delete_file called with unexpected ID")
        self.deleted_files.append(file_id)
        return True


class _StubIndexManager:
    def __init__(self):
        self.deleted_files: list[tuple[str, bool]] = []

    def delete_file_data(self, file_id: str, delete_file_metadata: bool = True):
        self.deleted_files.append((file_id, delete_file_metadata))
        return {
            "success": True,
            "file_id": file_id,
            "error_message": None,
        }


class _StubCfg:
    def __init__(self, storage: _StubFileStorage, index_manager: _StubIndexManager):
        class _FileStorageCfg:
            def __init__(self, instance):
                self._instance = instance

            def build(self):
                return self._instance

        class _IndexManagerCfg:
            def __init__(self, instance):
                self._instance = instance

            def build(self):
                return self._instance

        self.file_storage_config = _FileStorageCfg(storage)
        self.index_manager_config = _IndexManagerCfg(index_manager)
        self.max_concurrent_indexing = 1


class _CountingStorage:
    def __init__(self, counts):
        self.metadata_store = self
        self._counts = counts

    def count_accessible_files(self, user_id, status=None, search=None):  # noqa: ANN001
        return self._counts.get((user_id, status), 0)

    def list_accessible_files(self, user_id, status=None, limit=None, offset=None, search=None):  # noqa: ANN001
        return []

    def get_file_metadata(self, file_id: str):
        return None


class _CountingCfg:
    def __init__(self, storage):
        class _FileStorageCfg:
            def __init__(self, instance):
                self._instance = instance

            def build(self):
                return self._instance

        class _IndexManagerCfg:
            def build(self):
                return _StubIndexManager()

        self.file_storage_config = _FileStorageCfg(storage)
        self.index_manager_config = _IndexManagerCfg()
        self.max_concurrent_indexing = 1


@pytest.mark.asyncio
async def test_delete_file_marks_metadata_and_finishes_background_cleanup():
    owner_id = uuid.uuid4()
    metadata = _StubFileMetadata(file_id="file-123", owner_id=owner_id, status=FileStatus.INDEXED)
    storage = _StubFileStorage(metadata)
    index_manager = _StubIndexManager()
    knowledge = Knowledge(_StubCfg(storage, index_manager))

    result = await knowledge.delete_file(metadata.file_id, owner_id)

    assert result["status"] == "deleting"
    assert storage.status_updates[-1] == FileStatus.DELETED
    assert not knowledge.is_file_active(metadata.file_id)

    task = knowledge._active_deletion_tasks.get(metadata.file_id)
    assert task is not None, "background deletion task was not registered"
    await asyncio.wait_for(task, timeout=1)

    assert index_manager.deleted_files == [(metadata.file_id, True)]
    assert storage.deleted_files == [metadata.file_id]
    assert knowledge._is_file_marked_for_deletion(metadata.file_id) is False
    assert metadata.status == FileStatus.DELETED


def test_list_and_count_consistent_when_marked_for_deletion():
    owner_id = uuid.uuid4()
    metadata = _StubFileMetadata(file_id="file-xyz", owner_id=owner_id, status=FileStatus.STORED)
    storage = _StubFileStorage(metadata)
    index_manager = _StubIndexManager()
    knowledge = Knowledge(_StubCfg(storage, index_manager))

    knowledge._mark_file_for_deletion(metadata.file_id)

    files = knowledge.list_user_files(owner_id)
    assert files == []

    count = knowledge.count_user_files(owner_id)
    assert count == 0


def test_count_user_files_uses_owner_specific_marks():
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()
    storage = _CountingStorage({
        (owner_a, None): 10,
        (owner_a, FileStatus.DELETED): 2,
        (owner_b, None): 5,
        (owner_b, FileStatus.DELETED): 1,
    })
    knowledge = Knowledge(_CountingCfg(storage))

    knowledge._files_marked_for_deletion_by_owner[owner_a] = {"a1", "a2"}
    knowledge._files_marked_for_deletion_by_owner[owner_b] = {"b1"}

    assert knowledge.count_user_files(owner_a) == 10 - 2 - 2
    assert knowledge.count_user_files(owner_b) == 5 - 1 - 1
