from types import SimpleNamespace


def test_localdb_configured_base_path_uses_require_writable_dir(monkeypatch, tmp_path) -> None:
    # When base_path is explicitly configured, LocalDB must fail fast if it's not writable
    # rather than silently falling back to runtime directories (which breaks path consistency).
    import encapsulation.database.file_db.local as local_mod

    calls = {"require": 0, "ensure": 0}

    def _fake_require(path: str) -> str:
        calls["require"] += 1
        return str(tmp_path)

    def _fake_ensure(path: str, fallback_path: str | None = None) -> str:
        calls["ensure"] += 1
        return str(tmp_path)

    monkeypatch.setattr(local_mod, "require_writable_dir", _fake_require)
    monkeypatch.setattr(local_mod, "ensure_writable_dir", _fake_ensure)

    db = local_mod.LocalDB(SimpleNamespace(base_path=str(tmp_path)))
    _ = db._get_base_path()

    assert calls["require"] == 1
    assert calls["ensure"] == 0

