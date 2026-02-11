from typing import Literal

import pytest

from framework.config import AbstractConfig
from framework.register import Register


class _ExplodingConfig(AbstractConfig):
    type: Literal["explode"] = "explode"

    def build(self):
        raise ValueError("boom")


def test_get_object_missing_module_message_is_actionable():
    register = Register()
    register.registrations = {}
    register.registration_errors = {}

    with pytest.raises(KeyError) as exc:
        register.get_object("knowledge")

    msg = str(exc.value)
    assert "Module 'knowledge' is not registered" in msg
    assert "Available modules:" in msg


def test_get_object_includes_previous_registration_error(tmp_path):
    register = Register()
    register.registrations = {}
    register.registration_errors = {}

    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"type":"explode"}')

    with pytest.raises(ValueError):
        register.register(config_path=str(cfg), app_name="knowledge", config_type=_ExplodingConfig)

    with pytest.raises(KeyError) as exc:
        register.get_object("knowledge")

    msg = str(exc.value)
    assert "registration previously failed" in msg
    assert "boom" in msg

