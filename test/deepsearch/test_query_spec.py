import pytest

from core.deepsearch.query_spec import _coerce_query_spec, _get_target_langs


class TestCoerceQuerySpec:
    def test_minimal_valid(self):
        raw = {
            "report_needed": True,
            "report_style": "deepsearch",
            "bm25_terms": ["term1", "term2"],
            "regex_patterns": [],
            "reasoning": "Simple factual question.",
        }
        spec = _coerce_query_spec(raw, question="What is X?")
        assert spec["report_needed"] is True
        assert spec["bm25_terms"] == ["term1", "term2"]
        assert spec["report_style"] == "deepsearch"

    def test_empty_input_defaults(self):
        spec = _coerce_query_spec({}, question="Q")
        assert spec["report_needed"] is True
        assert spec["bm25_terms"] == []
        assert spec["regex_patterns"] == []

    def test_bm25_terms_max_10(self):
        raw = {"bm25_terms": [f"term{i}" for i in range(20)]}
        spec = _coerce_query_spec(raw, question="Q")
        assert len(spec["bm25_terms"]) == 10

    def test_regex_patterns_max_5(self):
        raw = {"regex_patterns": [f"pat{i}" for i in range(10)]}
        spec = _coerce_query_spec(raw, question="Q")
        assert len(spec["regex_patterns"]) == 5

    def test_multilang_terms(self):
        raw = {
            "bm25_terms": ["COSTCO"],
            "bm25_terms_by_lang": {
                "en": ["stockholders equity"],
                "zh-Hans": ["股东权益"],
            },
            "regex_patterns_by_lang": {
                "en": ["Total\\s+Equity"],
            },
        }
        spec = _coerce_query_spec(raw, question="Q")
        assert "en" in spec["bm25_terms_by_lang"]
        assert "zh-Hans" in spec["bm25_terms_by_lang"]
        assert spec["bm25_terms_by_lang"]["zh-Hans"] == ["股东权益"]

    def test_report_not_needed(self):
        raw = {"report_needed": False}
        spec = _coerce_query_spec(raw, question="What is GDP?")
        assert spec["report_needed"] is False


class TestGetTargetLangs:
    def test_empty_defaults_to_empty_list(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("QUERY_VARIANTS_LANGS", raising=False)
        assert _get_target_langs() == []

    def test_parses_comma_separated(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QUERY_VARIANTS_LANGS", "en, zh-Hans , ,fr")
        assert _get_target_langs() == ["en", "zh-Hans", "fr"]
