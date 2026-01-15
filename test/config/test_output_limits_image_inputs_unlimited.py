import runpy
from pathlib import Path

import pytest


def _eval_output_limits(monkeypatch: pytest.MonkeyPatch) -> dict:
    root = Path(__file__).resolve().parents[2]
    return runpy.run_path(str(root / "config" / "output_limits.py"))


def test_chat_and_deepsearch_max_image_inputs_leq_zero_means_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_ALL_EVIDENCE", raising=False)
    monkeypatch.setenv("CHAT_MAX_IMAGE_INPUTS", "0")
    monkeypatch.setenv("DEEPSEARCH_MAX_IMAGE_INPUTS", "-1")
    ns = _eval_output_limits(monkeypatch)
    assert ns["CHAT_MAX_IMAGE_INPUTS"] is None
    assert ns["DEEPSEARCH_MAX_IMAGE_INPUTS"] is None


def test_chat_and_deepsearch_max_image_inputs_positive_keeps_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_ALL_EVIDENCE", raising=False)
    monkeypatch.setenv("CHAT_MAX_IMAGE_INPUTS", "2")
    monkeypatch.setenv("DEEPSEARCH_MAX_IMAGE_INPUTS", "3")
    ns = _eval_output_limits(monkeypatch)
    assert ns["CHAT_MAX_IMAGE_INPUTS"] == 2
    assert ns["DEEPSEARCH_MAX_IMAGE_INPUTS"] == 3


def test_enable_all_evidence_always_unlimits_image_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ALL_EVIDENCE", "true")
    monkeypatch.setenv("CHAT_MAX_IMAGE_INPUTS", "2")
    monkeypatch.setenv("DEEPSEARCH_MAX_IMAGE_INPUTS", "3")
    ns = _eval_output_limits(monkeypatch)
    assert ns["CHAT_MAX_IMAGE_INPUTS"] is None
    assert ns["DEEPSEARCH_MAX_IMAGE_INPUTS"] is None

