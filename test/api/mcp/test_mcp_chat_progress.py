import uuid
import os
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


@pytest.mark.asyncio
async def test_mcp_chat_collects_progress_events(monkeypatch):
    import api.mcp.server as mcp_server

    user = SimpleNamespace(id=uuid.uuid4())
    session_id = str(uuid.uuid4())

    monkeypatch.setattr(mcp_server, "_safe_get_current_user_from_token", lambda _token: user)
    monkeypatch.setattr(mcp_server, "validate_user_session", lambda _session, _user: True)
    monkeypatch.setattr(mcp_server, "build_chat_evidence", lambda *a, **k: {"chunks": [{"id": uuid.uuid4()}]})

    class _SessionHandler:
        def get_session(self, _session_uuid):
            return SimpleNamespace(id=_session_uuid, user_id=user.id)

    class _MessageHandler:
        def create_message(self, _message):
            return True

    class _RAG:
        async def chat_async(self, query, owner_id, return_subgraph=True, progress_callback=None):  # noqa: ARG002
            assert owner_id == user.id
            if progress_callback:
                progress_callback({"stage": "retrieve", "status": "start", "trace": uuid.uuid4()})
                progress_callback({"stage": "retrieve", "status": "end"})
            return ("ok", [], {"nodes": []} if return_subgraph else None, {"meta": 1})

        def get_graph_store(self):
            return None

    class _Reg:
        def __init__(self):
            self._objects = {
                "chat_session": _SessionHandler(),
                "chat_message": _MessageHandler(),
                "rag_inference": _RAG(),
            }

        def get_object(self, key):
            return self._objects[key]

    monkeypatch.setattr(mcp_server, "registrator", _Reg())

    result = await mcp_server.chat.fn(session_id=session_id, query="hello", auth_token="token", ctx=None)

    assert result["session_id"] == session_id
    assert result["response"] == "ok"
    assert isinstance(result.get("progress"), list) and result["progress"]
    assert result["progress"][0]["stage"] == "retrieve"
    assert "evidence" in result
