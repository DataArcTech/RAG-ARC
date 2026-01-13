import os
import uuid
from typing import Optional, Union, Iterable, List

OwnerLike = Optional[Union[str, uuid.UUID]]
OwnerListLike = Optional[Iterable[OwnerLike]]


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


def _read_share_owner_id() -> Optional[str]:
    share_id = os.getenv("SHARE_OWNER_ID")
    if not share_id:
        return None
    share_id = share_id.strip()
    if not share_id:
        return None
    try:
        return str(uuid.UUID(share_id))
    except ValueError:
        return share_id


def _read_org_admin_owner_ids() -> List[str]:
    raw = os.getenv("ORG_ADMIN_OWNER_IDS")
    if not raw:
        return []
    tokens = []
    for item in str(raw).split(","):
        token = normalize_owner_id(item)
        if token:
            tokens.append(token)
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def is_admin_owner(owner_id: OwnerLike) -> bool:
    """
    Check whether the provided owner_id matches the ADMIN_OWNER_ID environment variable.
    """
    admin_id = _read_admin_owner_id()
    if not admin_id or owner_id is None:
        return False
    return str(owner_id) == admin_id


def is_org_admin_owner(owner_id: OwnerLike) -> bool:
    """
    Check whether owner_id is listed in ORG_ADMIN_OWNER_IDS (comma-separated UUIDs).
    """
    normalized = normalize_owner_id(owner_id)
    if not normalized:
        return False
    return normalized in set(_read_org_admin_owner_ids())


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


def normalize_owner_ids(owner_ids: OwnerListLike) -> List[str]:
    """
    Normalize and de-duplicate owner identifiers into canonical string form.

    Empty/None values are ignored.
    """
    if owner_ids is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for owner_id in owner_ids:
        token = normalize_owner_id(owner_id)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def get_admin_owner_id() -> Optional[str]:
    """Expose the normalized ADMIN_OWNER_ID value."""
    return _read_admin_owner_id()


def get_share_owner_id() -> Optional[str]:
    """Expose the normalized SHARE_OWNER_ID value."""
    return _read_share_owner_id()


def get_org_admin_owner_ids() -> List[str]:
    """Expose normalized ORG_ADMIN_OWNER_IDS values."""
    return _read_org_admin_owner_ids()


def is_global_admin_scope(owner_id: OwnerLike) -> bool:
    """Return True if the provided owner_id corresponds to ADMIN_OWNER_ID."""
    admin_id = _read_admin_owner_id()
    if not admin_id:
        return False
    normalized = normalize_owner_id(owner_id)
    return normalized == admin_id
