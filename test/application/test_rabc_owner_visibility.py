import uuid

import pytest

from application.rabc.visibility import build_owner_visibility


def test_build_owner_visibility_defaults_to_primary_only(monkeypatch):
    monkeypatch.delenv("SHARE_OWNER_ID", raising=False)
    monkeypatch.delenv("ADMIN_OWNER_ID", raising=False)

    owner_id = uuid.uuid4()
    vis = build_owner_visibility(primary_owner_id=owner_id)
    assert vis.owner_ids == (str(owner_id),)


def test_build_owner_visibility_includes_share_when_configured(monkeypatch):
    monkeypatch.setenv("SHARE_OWNER_ID", "00000000-0000-0000-0000-000000000002")
    monkeypatch.delenv("ADMIN_OWNER_ID", raising=False)

    owner_id = uuid.uuid4()
    vis = build_owner_visibility(primary_owner_id=owner_id, include_share=True)
    assert vis.owner_ids == (str(owner_id), "00000000-0000-0000-0000-000000000002")


def test_build_owner_visibility_rejects_share_equal_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_OWNER_ID", "00000000-0000-0000-0000-00000000ABCD")
    monkeypatch.setenv("SHARE_OWNER_ID", "00000000-0000-0000-0000-00000000ABCD")

    with pytest.raises(ValueError, match="SHARE_OWNER_ID must be different"):
        build_owner_visibility(primary_owner_id=str(uuid.uuid4()), include_share=True)

