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
    def __init__(self, files: list[_StubFileMetadata]):
        self._files = list(files)
        self.metadata_store = self

    def list_files(self, owner_id, status=None, limit=None, offset=None):  # noqa: ARG002
        files = [f for f in self._files if f.owner_id == owner_id]
        if status is not None:
            files = [f for f in files if f.status == status]
        return list(files)

    def list_accessible_files(self, user_id, status=None, limit=None, offset=None):  # noqa: ARG002
        return self.list_files(owner_id=user_id, status=status, limit=limit, offset=offset)

    def count_accessible_files(self, user_id, status=None):
        return len(self.list_accessible_files(user_id=user_id, status=status))

    def get_file_metadata(self, file_id: str):
        for f in self._files:
            if f.file_id == file_id:
                return f
        return None


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
async def test_list_visible_files_unions_me_and_share_when_enabled():
    me = uuid.uuid4()
    share = uuid.uuid4()

    files = [
        _StubFileMetadata(file_id="file-me", owner_id=me, status=FileStatus.STORED, created_at=1.0),
        _StubFileMetadata(file_id="file-share", owner_id=share, status=FileStatus.STORED, created_at=2.0),
    ]
    storage = _StubFileStorage(files)
    knowledge = Knowledge(_StubCfg(storage))

    result = knowledge.list_visible_files(me, include_share=True, share_owner_id=share)
    assert [f.file_id for f in result] == ["file-share", "file-me"]

    result_async = await knowledge.list_visible_files_async(me, include_share=True, share_owner_id=share)
    assert [f.file_id for f in result_async] == ["file-share", "file-me"]

    assert knowledge.count_visible_files(me, include_share=True, share_owner_id=share) == 2
    assert await knowledge.count_visible_files_async(me, include_share=True, share_owner_id=share) == 2

