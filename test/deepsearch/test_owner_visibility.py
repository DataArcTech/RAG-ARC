import pytest

from core.deepsearch.utils.owner_visibility import resolve_owner_visibility
from core.graph_adapter.base import GraphAccessScope


def test_owner_visibility_defaults_to_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARE_OWNER_ID", raising=False)
    res = resolve_owner_visibility(extra={}, access_scope=GraphAccessScope(scope_id="owner-1"), graph_context_metadata={})
    assert res.primary_owner_id == "owner-1"
    assert res.owner_ids_used == ("owner-1",)
    assert res.owner_ids_rejected == ()


def test_owner_visibility_allows_me_plus_share(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    res = resolve_owner_visibility(
        extra={"owner_ids": ["owner-1", "share-1"]},
        access_scope=GraphAccessScope(scope_id="owner-1"),
        graph_context_metadata={},
    )
    assert res.owner_ids_used == ("owner-1", "share-1")
    assert res.owner_ids_rejected == ()


def test_owner_visibility_rejects_unknown_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARE_OWNER_ID", "share-1")
    res = resolve_owner_visibility(
        extra={"owner_ids": ["owner-1", "share-1", "other-1"]},
        access_scope=GraphAccessScope(scope_id="owner-1"),
        graph_context_metadata={},
    )
    assert res.owner_ids_used == ("owner-1", "share-1")
    assert res.owner_ids_rejected == ("other-1",)


def test_owner_visibility_admin_allows_any_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_OWNER_ID", "admin-1")
    res = resolve_owner_visibility(
        extra={"owner_ids": ["owner-1", "share-1", "other-1"]},
        access_scope=GraphAccessScope(scope_id="admin-1"),
        graph_context_metadata={},
    )
    assert res.primary_owner_id == "admin-1"
    assert res.owner_ids_used == ("owner-1", "share-1", "other-1")
    assert res.owner_ids_rejected == ()

