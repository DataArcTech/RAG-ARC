import pytest

from framework.register import Register


def test_substitute_env_vars_full_placeholder_unset_becomes_none(monkeypatch):
    monkeypatch.delenv("DEEPSEARCH_TOOL_ARTIFACT_DIR", raising=False)
    register = Register()
    resolved = register._substitute_env_vars(
        {"artifact_dir": "${DEEPSEARCH_TOOL_ARTIFACT_DIR}"}
    )
    assert "artifact_dir" not in resolved


def test_substitute_env_vars_full_placeholder_set_resolves_value(monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_TOOL_ARTIFACT_DIR", "./local/deepsearch_artifacts")
    register = Register()
    resolved = register._substitute_env_vars(
        {"artifact_dir": "${DEEPSEARCH_TOOL_ARTIFACT_DIR}"}
    )
    assert resolved["artifact_dir"] == "./local/deepsearch_artifacts"


def test_substitute_env_vars_embedded_placeholder_unset_keeps_string(monkeypatch):
    monkeypatch.delenv("SOME_MISSING_VAR", raising=False)
    register = Register()
    resolved = register._substitute_env_vars(
        {"path": "prefix/${SOME_MISSING_VAR}/suffix"}
    )
    assert resolved["path"] == "prefix/${SOME_MISSING_VAR}/suffix"
