import os
import uuid
from types import SimpleNamespace

from api.utils.owner_scope import resolve_default_owner_id


def test_chatbot_uses_shared_owner_for_chatkb_users():
    shared_owner_id = uuid.uuid4()
    os.environ["CHATBOT_SHARED_DOCUMENT_OWNER_ID"] = str(shared_owner_id)
    user = SimpleNamespace(id=uuid.uuid4(), type=1)
    assert resolve_default_owner_id(user) == shared_owner_id


def test_chatbot_uses_user_owner_for_livingkb_users():
    shared_owner_id = uuid.uuid4()
    os.environ["CHATBOT_SHARED_DOCUMENT_OWNER_ID"] = str(shared_owner_id)
    user = SimpleNamespace(id=uuid.uuid4(), type=0)
    assert resolve_default_owner_id(user) == user.id

