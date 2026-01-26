import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add project root to path before any imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _project_root)

# Set environment variables that conftest.py would set
os.environ.setdefault("TASK_QUEUE_MODE", "inprocess")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
os.environ.setdefault("RAGARC_INDEXING_DEPENDENCY_CHECK_MODE", "off")
os.environ.setdefault("KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS", "0")


@pytest.fixture
def client(monkeypatch):
    import api.routers.knowledge as knowledge_router

    user = SimpleNamespace(id=uuid.uuid4())
    calls: dict[str, object] = {}

    class _StubKnowledge:
        async def upload_file(self, file, user_id, *, relative_path=None):  # noqa: ANN001
            calls["upload"] = (file.filename, user_id, relative_path)
            return "doc-1"

        async def list_user_files_async(self, user_id, status=None, limit=None, offset=None, search=None):  # noqa: ANN001
            calls["list"] = (user_id, status, limit, offset, search)
            return []

        async def count_user_files_async(self, user_id, status=None, search=None):  # noqa: ANN001
            calls["count"] = (user_id, status, search)
            return 0

    monkeypatch.setattr(knowledge_router, "get_knowledge_handler", lambda: _StubKnowledge())

    app = FastAPI()
    app.include_router(knowledge_router.router)
    app.dependency_overrides[knowledge_router.get_current_user] = lambda: user
    client = TestClient(app)
    client._calls = calls  # type: ignore[attr-defined]
    client._user = user  # type: ignore[attr-defined]
    return client


def test_knowledge_upload_passes_relative_path(client, tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello", encoding="utf-8")
    with file_path.open("rb") as f:
        resp = client.post(
            "/knowledge",
            files={"file": ("a.txt", f, "text/plain")},
            data={"relative_path": "docs/a.txt"},
        )
    assert resp.status_code == 200
    assert resp.json() == "doc-1"
    filename, user_id, rel = client._calls["upload"]  # type: ignore[attr-defined]
    assert filename == "a.txt"
    assert user_id == client._user.id  # type: ignore[attr-defined]
    assert rel == "docs/a.txt"


def test_knowledge_list_uses_async_wrappers(client):
    resp = client.get("/knowledge/list_files?pagesize=50&page=1")
    assert resp.status_code == 200
    assert resp.json()["files"] == []
    assert resp.json()["total"] == 0
    user_id, status, limit, offset, search = client._calls["list"]  # type: ignore[attr-defined]
    assert user_id == client._user.id  # type: ignore[attr-defined]
    assert status is None
    assert limit == 50
    assert offset == 0
    assert search is None


def test_knowledge_get_chunk_returns_chunk_content(monkeypatch):
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

    monkeypatch.setattr(knowledge_router, "get_knowledge_handler", lambda: _StubKnowledge())
    monkeypatch.setattr(knowledge_router.registrator, "get_object", lambda name: _StubRag() if name == "rag_inference" else object())

    app = FastAPI()
    app.include_router(knowledge_router.router)
    app.dependency_overrides[knowledge_router.get_current_user] = lambda: user
    client = TestClient(app)

    resp = client.get("/knowledge/chunk/rep-0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_id"] == "rep-0"
    assert body["content"] == "Display content"
    assert body["file_id"] == "file-0"
    assert body["filename"] == "doc0.md"
    assert body["document_url"] == "/knowledge/file-0/download"


def test_knowledge_upload_file_validation(monkeypatch):
    """测试文件上传验证：文件格式和文件大小限制"""
    import api.routers.knowledge as knowledge_router
    
    user = SimpleNamespace(id=uuid.uuid4())
    
    class _StubKnowledge:
        async def upload_file(self, file, user_id, *, relative_path=None):  # noqa: ANN001
            return "doc-1"
    
    monkeypatch.setattr(knowledge_router, "get_knowledge_handler", lambda: _StubKnowledge())
    
    app = FastAPI()
    app.include_router(knowledge_router.router)
    app.dependency_overrides[knowledge_router.get_current_user] = lambda: user
    test_client = TestClient(app)
    
    # ========== 文件格式测试 ==========
    allowed_extensions = {'.docx', '.xlsx', '.pptx', '.pdf', '.jpg', '.jpeg', '.png', '.txt', '.html', '.md'}
    
    # 测试1: 支持的文件格式应该成功（测试所有支持的类型）
    for ext in allowed_extensions:
        resp = test_client.post(
            "/knowledge",
            files={"file": (f"test{ext}", b"test content", "application/octet-stream")},
        )
        assert resp.status_code == 200, f"支持的文件格式 {ext} 应该成功，但返回了 {resp.status_code}: {resp.text}"
    
    # 测试2: 不支持的文件格式应该失败
    unsupported_extensions = ['.exe', '.zip', '.rar', '.mp4', '.py', '.js']
    for ext in unsupported_extensions:
        resp = test_client.post(
            "/knowledge",
            files={"file": (f"test{ext}", b"test content", "application/octet-stream")},
        )
        assert resp.status_code == 400, f"不支持的文件格式 {ext} 应该失败，但返回了 {resp.status_code}"
        assert "不支持的文件类型" in resp.json()["detail"], f"错误消息应该包含'不支持的文件类型'，但得到: {resp.json()['detail']}"
    
    # 测试3: 无扩展名的文件应该失败
    resp = test_client.post(
        "/knowledge",
        files={"file": ("testfile", b"test content", "application/octet-stream")},
    )
    assert resp.status_code == 400, "无扩展名的文件应该失败"
    assert "不支持的文件类型" in resp.json()["detail"]
    
    # ========== 文件大小测试 ==========
    # 测试4: 小于10MB的文件应该成功
    small_file_size = 5 * 1024 * 1024  # 5MB
    small_file_content = b"x" * small_file_size
    resp = test_client.post(
        "/knowledge",
        files={"file": ("test_small.txt", small_file_content, "text/plain")},
    )
    assert resp.status_code == 200, f"小于10MB的文件应该成功，但返回了 {resp.status_code}: {resp.text}"
    
    # 测试5: 正好10MB的文件应该成功（边界情况）
    exact_file_size = 10 * 1024 * 1024  # 10MB
    exact_file_content = b"x" * exact_file_size
    resp = test_client.post(
        "/knowledge",
        files={"file": ("test_exact.txt", exact_file_content, "text/plain")},
    )
    assert resp.status_code == 200, f"正好10MB的文件应该成功，但返回了 {resp.status_code}: {resp.text}"
    
    # 测试6: 大于10MB的文件应该失败
    large_file_size = 11 * 1024 * 1024  # 11MB
    large_file_content = b"x" * large_file_size
    resp = test_client.post(
        "/knowledge",
        files={"file": ("test_large.txt", large_file_content, "text/plain")},
    )
    assert resp.status_code == 400, f"大于10MB的文件应该失败，但返回了 {resp.status_code}"
    assert "该文件超过10MB，请检查后重新上传！" == resp.json()["detail"], f"错误消息应该是'该文件超过10MB，请检查后重新上传！'，但得到: {resp.json()['detail']}"
