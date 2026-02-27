"""
Owner selection tests for RAG SSE routing.

These are unit-level to avoid ASGI streaming close semantics in the sandbox.
"""

import uuid
from types import SimpleNamespace


def test_stream_chat_livingkb_uses_user_id_as_owner():
    from api.routers.rag_inference_handlers import get_default_owner_id
    from api.routers.rag_inference_modules.stream_chat.utils.owner_resolver import resolve_effective_owner

    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, type=0)  # livingKB user

    assert get_default_owner_id(user) == user_id
    assert resolve_effective_owner(user, target_owner_id=None, include_all_owners=False) == user_id


def test_stream_chat_chatkb_uses_shared_owner_id(monkeypatch):
    from api.routers import rag_inference_handlers
    from api.routers.rag_inference_handlers import get_default_owner_id
    from api.routers.rag_inference_modules.stream_chat.utils.owner_resolver import resolve_effective_owner

    user_id = uuid.uuid4()
    shared_owner_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, type=1)  # chatKB user

    monkeypatch.setattr(rag_inference_handlers, "get_shared_document_owner_id", lambda: shared_owner_id)

    assert get_default_owner_id(user) == shared_owner_id
    assert resolve_effective_owner(user, target_owner_id=None, include_all_owners=False) == shared_owner_id


def test_stream_chat_default_type_uses_user_id():
    from api.routers.rag_inference_handlers import get_default_owner_id
    from api.routers.rag_inference_modules.stream_chat.utils.owner_resolver import resolve_effective_owner

    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id)  # default type=0

    assert get_default_owner_id(user) == user_id
    assert resolve_effective_owner(user, target_owner_id=None, include_all_owners=False) == user_id

