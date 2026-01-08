import json

from core.graph_adapter.hipporag_scope_helpers import maybe_rewrite_query_for_scope


class _StubLLM:
    def __init__(self, *, zh: str = "", en: str = "") -> None:
        self._zh = zh
        self._en = en

    def chat(self, messages, **kwargs):  # noqa: ANN001,ARG002
        return json.dumps({"zh_hans": self._zh, "zh_hant": "", "en": self._en})


def test_xlang_rewrite_defaults_trigger_en_to_zh(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_ZH_MIN", raising=False)
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_ZH_MAX", raising=False)
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_EN_MIN", raising=False)
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_EN_MAX", raising=False)

    rewritten = maybe_rewrite_query_for_scope(llm_client=_StubLLM(zh="中文改写"), query="hello world")
    assert rewritten is not None
    assert "中文改写" in rewritten


def test_xlang_rewrite_thresholds_can_disable_en_to_zh(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_ZH_MIN", "0.30")
    monkeypatch.setenv("DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_ZH_MAX", "0.05")
    # 3 alpha chars out of 12 => 0.25 (below 0.30), should not rewrite.
    rewritten = maybe_rewrite_query_for_scope(llm_client=_StubLLM(zh="中文改写"), query="abc123456789")
    assert rewritten is None


def test_xlang_rewrite_defaults_trigger_zh_to_en(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_ZH_MIN", raising=False)
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_ZH_MAX", raising=False)
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_EN_MIN", raising=False)
    monkeypatch.delenv("DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_EN_MAX", raising=False)

    rewritten = maybe_rewrite_query_for_scope(llm_client=_StubLLM(en="english rewrite"), query="保险条款")
    assert rewritten is not None
    assert "english rewrite" in rewritten

