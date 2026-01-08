import os
import uuid
from typing import Protocol


class UserLike(Protocol):
    id: uuid.UUID
    type: int | None


def get_shared_document_owner_id() -> uuid.UUID:
    raw = os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001")
    try:
        return uuid.UUID(str(raw))
    except ValueError as exc:
        raise RuntimeError("CHATBOT_SHARED_DOCUMENT_OWNER_ID must be a valid UUID") from exc


def resolve_default_owner_id(current_user: UserLike) -> uuid.UUID:
    """Resolve the default retrieval owner scope for the given user.

    - type=1 (chatKB): use the shared document owner ID so all chatKB users share one knowledge space.
    - type=0 (livingKB): use the authenticated user's ID for tenant isolation.
    """

    raw_type = getattr(current_user, "type", 0)
    try:
        user_type = int(raw_type) if raw_type is not None else 0
    except (TypeError, ValueError):
        user_type = 0

    if user_type == 1:
        return get_shared_document_owner_id()
    return current_user.id

