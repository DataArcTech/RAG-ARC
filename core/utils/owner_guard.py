import os
import uuid
from typing import Optional, Union

OwnerLike = Optional[Union[str, uuid.UUID]]


def _read_admin_owner_id() -> Optional[str]:
    admin_id = os.getenv("ADMIN_OWNER_ID")
    if not admin_id:
        return None
    admin_id = admin_id.strip()
    if not admin_id:
        return None
    try:
        return str(uuid.UUID(admin_id))
    except ValueError:
        return admin_id


def is_admin_owner(owner_id: OwnerLike) -> bool:
    """
    Check whether the provided owner_id matches the ADMIN_OWNER_ID environment variable.
    """
    admin_id = _read_admin_owner_id()
    if not admin_id or owner_id is None:
        return False
    return str(owner_id) == admin_id


def normalize_owner_id(owner_id: OwnerLike) -> Optional[str]:
    """
    Normalize owner identifiers into strings when possible (helper for logging/tests).
    """
    if owner_id is None:
        return None
    try:
        return str(uuid.UUID(str(owner_id)))
    except ValueError:
        return str(owner_id)
