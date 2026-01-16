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


class _StubIndexManager:
    def delete_file_data(self, file_id: str, delete_file_metadata: bool = True):  # noqa: ARG002
        return {"success": True, "file_id": file_id, "error_message": None}


class _StubCfg:
    def __init__(self, storage: _StubFileStorage):
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
async def test_async_wrappers_match_sync_behavior_for_marked_deletion():
    owner_id = uuid.uuid4()
    metadata = _StubFileMetadata(file_id="file-1", owner_id=owner_id, status=FileStatus.STORED)
    storage = _StubFileStorage(metadata)
    knowledge = Knowledge(_StubCfg(storage))

    knowledge._mark_file_for_deletion(metadata.file_id)

    assert knowledge.list_user_files(owner_id) == []
    assert knowledge.count_user_files(owner_id) == 0

    files = await knowledge.list_user_files_async(owner_id)
    count = await knowledge.count_user_files_async(owner_id)
    assert files == []
    assert count == 0
