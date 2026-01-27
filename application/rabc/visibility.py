import logging
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from core.utils.owner_guard import get_admin_owner_id, get_share_owner_id, normalize_owner_id, normalize_owner_ids

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OwnerVisibility:
    """
    Owner-based visibility scope for algorithm layer.

    - owner_ids: normalized owner_id strings to include in reads/retrieval.
    - label: human-readable label for logging/diagnostics.
    """

    owner_ids: tuple[str, ...]
    label: str


def _coerce_owner_token(owner_id: uuid.UUID | str | None) -> Optional[str]:
    return normalize_owner_id(owner_id)


def resolve_share_owner_token(share_owner_id: uuid.UUID | str | None = None) -> Optional[str]:
    if share_owner_id is not None:
        return _coerce_owner_token(share_owner_id)
    return get_share_owner_id()


def build_owner_visibility(
    *,
    primary_owner_id: uuid.UUID | str,
    include_share: bool = False,
    share_owner_id: uuid.UUID | str | None = None,
    owner_ids: Sequence[uuid.UUID | str] | None = None,
    label: str | None = None,
) -> OwnerVisibility:
    """
    Build the owner visibility scope for reads/retrieval.

    Rules:
    - Default: only include primary owner ("me" / "target_user").
    - include_share=True: include both primary + share owner.

    Callers may provide `owner_ids` directly to fully control visibility; when provided,
    `include_share` is applied on top of it (i.e., union).
    """
    primary_token = _coerce_owner_token(primary_owner_id)
    if not primary_token:
        raise ValueError("primary_owner_id must be a non-empty UUID/string")

    visible = list(normalize_owner_ids(owner_ids) if owner_ids is not None else [primary_token])
    if primary_token not in visible:
        visible.insert(0, primary_token)

    if include_share:
        share_token = resolve_share_owner_token(share_owner_id)
        if not share_token:
            logger.warning("include_share requested but SHARE_OWNER_ID is not configured; falling back to primary only")
        else:
            admin_token = get_admin_owner_id()
            if admin_token and share_token == admin_token:
                raise ValueError("Invalid configuration: SHARE_OWNER_ID must be different from ADMIN_OWNER_ID")
            if share_token not in visible:
                visible.append(share_token)

    scope_label = label
    if not scope_label:
        scope_label = "+".join(visible)
    return OwnerVisibility(owner_ids=tuple(visible), label=scope_label)

