import os
import sys
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

# Add project root to path before any imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _project_root)

# Set environment variables that conftest.py would set
os.environ.setdefault("TASK_QUEUE_MODE", "inprocess")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
os.environ.setdefault("RAGARC_INDEXING_DEPENDENCY_CHECK_MODE", "off")
os.environ.setdefault("KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS", "0")


class _ASGIClient:
    def __init__(self, *, app: FastAPI, http: httpx.AsyncClient, calls: dict[str, object], user: SimpleNamespace):
        self.app = app
        self.http = http
        self._calls = calls
        self._user = user

    async def get(self, *args, **kwargs):  # noqa: ANN001, D401
        return await self.http.get(*args, **kwargs)

    async def post(self, *args, **kwargs):  # noqa: ANN001, D401
        return await self.http.post(*args, **kwargs)


@pytest.fixture
async def client(monkeypatch):
    import api.routers.knowledge as knowledge_router

    user = SimpleNamespace(id=uuid.uuid4())
    calls: dict[str, object] = {}

    class _StubKnowledge:
        async def upload_file(self, file, user_id, *, relative_path=None, ingest_mode="index"):  # noqa: ANN001
            calls["upload"] = (file.filename, user_id, relative_path, ingest_mode)
            return "doc-1"

        async def list_user_files_async(self, user_id, status=None, limit=None, offset=None, search=None):  # noqa: ANN001
            calls["list"] = (user_id, status, limit, offset, search)
            return []

        async def count_user_files_async(self, user_id, status=None, search=None):  # noqa: ANN001
            calls["count"] = (user_id, status, search)
            return 0

    def _get_stub_knowledge():  # noqa: ANN001
        return _StubKnowledge()

    monkeypatch.setattr(knowledge_router, "get_knowledge_handler", _get_stub_knowledge)

    app = FastAPI()
    app.include_router(knowledge_router.router)
    async def _stub_user():  # noqa: ANN001
        return user

    app.dependency_overrides[knowledge_router.get_current_user] = _stub_user
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield _ASGIClient(app=app, http=http, calls=calls, user=user)


async def test_knowledge_upload_passes_relative_path(client, tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello", encoding="utf-8")
    with file_path.open("rb") as f:
        resp = await client.post(
            "/knowledge",
            files={"file": ("a.txt", f, "text/plain")},
            data={"relative_path": "docs/a.txt"},
        )
    assert resp.status_code == 200
    assert resp.json() == "doc-1"
    filename, user_id, rel, ingest_mode = client._calls["upload"]
    assert filename == "a.txt"
    assert user_id == client._user.id
    assert rel == "docs/a.txt"
    assert ingest_mode == "index"


async def test_knowledge_list_uses_async_wrappers(client):
    resp = await client.get("/knowledge/list_files?pagesize=50&page=1")
    assert resp.status_code == 200
    assert resp.json()["files"] == []
    assert resp.json()["total"] == 0
    user_id, status, limit, offset, search = client._calls["list"]
    assert user_id == client._user.id
    assert status is None
    assert limit == 50
    assert offset == 0
    assert search is None


async def test_knowledge_get_chunk_returns_chunk_content(monkeypatch):
    import api.routers.knowledge as knowledge_router

    user = SimpleNamespace(id=uuid.uuid4())

    class _StubFileMeta:
        filename = "doc0.md"

    class _StubFileStorage:
        def get_file_metadata(self, _file_id):  # noqa: ANN001
            return _StubFileMeta()

    class _StubKnowledge:
        file_storage = _StubFileStorage()

        def check_file_access(self, _file_id, _user_id):  # noqa: ANN001
            return "view"

    class _StubChunk:
        id = "rep-0"
        content = "Chunk content"
        metadata = {"source_file_id": "file-0", "prompt_text": "Display content"}

    class _StubGraphStore:
        def get_by_ids(self, ids):  # noqa: ANN001
            return [_StubChunk()] if ids and ids[0] == "rep-0" else []

    class _StubRag:
        def get_graph_store(self):  # noqa: ANN201
            return _StubGraphStore()

    def _get_stub_knowledge():  # noqa: ANN001
        return _StubKnowledge()

    monkeypatch.setattr(knowledge_router, "get_knowledge_handler", _get_stub_knowledge)
    monkeypatch.setattr(knowledge_router.registrator, "get_object", lambda name: _StubRag() if name == "rag_inference" else object())

    app = FastAPI()
    app.include_router(knowledge_router.router)
    async def _stub_user():  # noqa: ANN001
        return user

    app.dependency_overrides[knowledge_router.get_current_user] = _stub_user
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/knowledge/chunk/rep-0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_id"] == "rep-0"
    assert body["content"] == "Display content"
    assert body["file_id"] == "file-0"
    assert body["filename"] == "doc0.md"
    assert body["document_url"] == "/knowledge/file-0/download"


def test_knowledge_upload_file_validation():
    """测试文件上传验证：文件格式和文件大小限制（不走 HTTP，避免大文件 multipart 造成 CI 超时）"""

    from fastapi import HTTPException

    import api.routers.knowledge as knowledge_router

    # ========== 文件格式测试 ==========
    for filename in ("test.pdf", "test.md", "test.png"):
        knowledge_router._validate_upload_extension(filename)

    with pytest.raises(HTTPException) as exc:
        knowledge_router._validate_upload_extension("test.exe")
    assert exc.value.status_code == 400
    assert "不支持的文件类型" in str(exc.value.detail)

    with pytest.raises(HTTPException) as exc:
        knowledge_router._validate_upload_extension("testfile")
    assert exc.value.status_code == 400
    assert "不支持的文件类型" in str(exc.value.detail)

    # ========== 文件大小测试 ==========
    knowledge_router._validate_upload_size(1024)

    with pytest.raises(HTTPException) as exc:
        knowledge_router._validate_upload_size(int(knowledge_router.MAX_UPLOAD_BYTES) + 1)
    assert exc.value.status_code == 400
    assert str(exc.value.detail) == "该文件超过10MB，请检查后重新上传！"
