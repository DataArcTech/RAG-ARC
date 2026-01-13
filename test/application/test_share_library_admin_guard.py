import io
import uuid
from dataclasses import dataclass

import pytest
from fastapi import HTTPException

from application.knowledge.module import Knowledge
from encapsulation.data_model.orm_models import FileStatus


class _DummyUpload:
    def __init__(self, filename: str = "x.txt", content: bytes = b"hello"):
        self.filename = filename
        self.content_type = "text/plain"
        self.file = io.BytesIO(content)


@dataclass
class _StubFileMetadata:
    file_id: str
    owner_id: uuid.UUID
    status: FileStatus
    filename: str = "stub.txt"
    blob_key: str = "blob-key"
    file_size: int = 0
    content_type: str = "text/plain"
    created_at: float = 0.0
    updated_at: float = 0.0


class _StubFileStorage:
    def __init__(self, metadata: _StubFileMetadata):
        self._metadata = metadata
        self.metadata_store = self
        self.upload_calls: list[dict] = []

    def upload_file(self, *, filename, file_data, owner_id, content_type):  # noqa: ARG002
        self.upload_calls.append(
            {"filename": filename, "owner_id": owner_id, "content_type": content_type, "bytes": len(file_data or b"")}
        )
        return "file-1"

    def get_file_metadata(self, file_id: str):  # noqa: ARG002
        return self._metadata

    def update_file_status(self, file_id: str, status: FileStatus):  # noqa: ARG002
        return True


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
async def test_share_library_upload_requires_admin(monkeypatch):
    share_owner = uuid.uuid4()
    monkeypatch.setenv("SHARE_OWNER_ID", str(share_owner))
    monkeypatch.setenv("ADMIN_OWNER_ID", str(uuid.uuid4()))
    monkeypatch.delenv("ORG_ADMIN_OWNER_IDS", raising=False)

    storage = _StubFileStorage(_StubFileMetadata(file_id="file-1", owner_id=share_owner, status=FileStatus.STORED))
    knowledge = Knowledge(_StubCfg(storage))

    with pytest.raises(HTTPException) as exc:
        await knowledge.upload_file_scoped(
            file=_DummyUpload(),
            actor_id=uuid.uuid4(),
            owner_id=share_owner,
            allow_non_owner=True,
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_share_library_upload_allows_admin_owner_id(monkeypatch):
    share_owner = uuid.uuid4()
    admin = uuid.uuid4()
    monkeypatch.setenv("SHARE_OWNER_ID", str(share_owner))
    monkeypatch.setenv("ADMIN_OWNER_ID", str(admin))
    monkeypatch.delenv("ORG_ADMIN_OWNER_IDS", raising=False)

    storage = _StubFileStorage(_StubFileMetadata(file_id="file-1", owner_id=share_owner, status=FileStatus.STORED))
    knowledge = Knowledge(_StubCfg(storage))

    file_id = await knowledge.upload_file_scoped(
        file=_DummyUpload(),
        actor_id=admin,
        owner_id=share_owner,
        allow_non_owner=True,
    )
    assert file_id == "file-1"
    assert storage.upload_calls and storage.upload_calls[0]["owner_id"] == share_owner


@pytest.mark.asyncio
async def test_share_library_upload_allows_org_admin_allowlist(monkeypatch):
    share_owner = uuid.uuid4()
    org_admin = uuid.uuid4()
    monkeypatch.setenv("SHARE_OWNER_ID", str(share_owner))
    monkeypatch.delenv("ADMIN_OWNER_ID", raising=False)
    monkeypatch.setenv("ORG_ADMIN_OWNER_IDS", str(org_admin))

    storage = _StubFileStorage(_StubFileMetadata(file_id="file-1", owner_id=share_owner, status=FileStatus.STORED))
    knowledge = Knowledge(_StubCfg(storage))

    file_id = await knowledge.upload_file_scoped(
        file=_DummyUpload(),
        actor_id=org_admin,
        owner_id=share_owner,
        allow_non_owner=True,
    )
    assert file_id == "file-1"


@pytest.mark.asyncio
async def test_share_library_delete_requires_admin(monkeypatch):
    share_owner = uuid.uuid4()
    monkeypatch.setenv("SHARE_OWNER_ID", str(share_owner))
    monkeypatch.delenv("ADMIN_OWNER_ID", raising=False)
    monkeypatch.delenv("ORG_ADMIN_OWNER_IDS", raising=False)

    storage = _StubFileStorage(_StubFileMetadata(file_id="file-1", owner_id=share_owner, status=FileStatus.STORED))
    knowledge = Knowledge(_StubCfg(storage))

    with pytest.raises(HTTPException) as exc:
        await knowledge.delete_file_scoped(
            "file-1",
            actor_id=uuid.uuid4(),
            owner_id=share_owner,
            allow_non_owner=True,
        )
    assert exc.value.status_code == 403
