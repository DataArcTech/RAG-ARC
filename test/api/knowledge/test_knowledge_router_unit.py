import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


@pytest.fixture
def client(monkeypatch):
    import api.routers.knowledge as knowledge_router

    user = SimpleNamespace(id=uuid.uuid4())
    calls: dict[str, object] = {}

    class _StubKnowledge:
        async def upload_file(self, file, user_id, *, relative_path=None):  # noqa: ANN001
            calls["upload"] = (file.filename, user_id, relative_path)
            return "doc-1"

        async def list_user_files_async(self, user_id, status=None, limit=None, offset=None):  # noqa: ANN001
            calls["list"] = (user_id, status, limit, offset)
            return []

        async def count_user_files_async(self, user_id, status=None):  # noqa: ANN001
            calls["count"] = (user_id, status)
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
    resp = client.get("/knowledge/list_files?limit=50&offset=0")
    assert resp.status_code == 200
    assert resp.json()["files"] == []
    assert resp.json()["total"] == 0
    user_id, status, limit, offset = client._calls["list"]  # type: ignore[attr-defined]
    assert user_id == client._user.id  # type: ignore[attr-defined]
    assert status is None
    assert limit == 50
    assert offset == 0


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
