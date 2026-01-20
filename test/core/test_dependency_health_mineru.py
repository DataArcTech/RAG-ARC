import types

import core.utils.dependency_health as dh


def test_check_mineru_ok(monkeypatch) -> None:
    monkeypatch.setenv("MINERU_SERVER_URL", "http://127.0.0.1:8899")

    class _Resp:
        status_code = 200

        def json(self):
            return {"status": "healthy"}

    def _fake_get(url: str, timeout: float):
        assert url == "http://127.0.0.1:8899/health"
        assert timeout > 0
        return _Resp()

    monkeypatch.setattr(dh, "httpx", types.SimpleNamespace(get=_fake_get))

    res = dh.check_mineru()
    assert res["ok"] is True


def test_check_dependencies_includes_mineru_when_parse_mode_mineru(monkeypatch) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "mineru")
    monkeypatch.setenv("MINERU_SERVER_URL", "http://127.0.0.1:8899")

    monkeypatch.setattr(dh, "check_mineru", lambda: {"ok": True})
    health = dh.check_dependencies(
        include_postgres=False,
        include_redis=False,
        include_neo4j=False,
        mode_env="RAGARC_DEPENDENCY_CHECK_MODE",
        default_mode="warn",
    )
    assert "mineru" in (health.get("checks") or {})

