import pytest

from core.deepsearch.query_spec import _coerce_query_spec, _get_target_langs


class TestCoerceQuerySpec:
    def test_minimal_valid(self):
        raw = {
            "question_kind": "factual",
            "is_computable": False,
            "report_needed": True,
            "report_style": "deepsearch",
            "bm25_terms": ["term1", "term2"],
            "regex_patterns": [],
            "hyde_query": "The document says...",
            "compute_spec": None,
            "reasoning": "Simple factual question.",
        }
        spec = _coerce_query_spec(raw, question="What is X?")
        assert spec["question_kind"] == "factual"
        assert spec["is_computable"] is False
        assert spec["bm25_terms"] == ["term1", "term2"]
        assert spec["hyde_query"] == "The document says..."
        assert spec["compute_spec"] is None

    def test_computable_with_compute_spec(self):
        raw = {
            "question_kind": "computable",
            "is_computable": True,
            "report_needed": True,
            "report_style": "deepsearch",
            "bm25_terms": ["ROCE", "return on capital"],
            "regex_patterns": ["Total\\s+Assets", "FY\\s*2023"],
            "hyde_query": "ROCE = net income / capital employed = ...",
            "compute_spec": {
                "formula_hint": "ROCE = net_income / capital_employed",
                "required_fields": [
                    {"name": "net_income", "hint": "Net Income"},
                    {"name": "capital_employed", "hint": "Capital Employed"},
                ],
                "rounding": "3 decimal places",
            },
            "reasoning": "User asks for ROCE ratio.",
        }
        spec = _coerce_query_spec(raw, question="What is the ROCE?")
        assert spec["is_computable"] is True
        assert spec["compute_spec"] is not None
        assert len(spec["compute_spec"]["required_fields"]) == 2
        assert spec["compute_spec"]["formula_hint"] == "ROCE = net_income / capital_employed"

    def test_empty_input_defaults(self):
        spec = _coerce_query_spec({}, question="Q")
        assert spec["question_kind"] == "factual"
        assert spec["is_computable"] is False
        assert spec["bm25_terms"] == []
        assert spec["regex_patterns"] == []
        assert spec["compute_spec"] is None

    def test_invalid_question_kind_defaults(self):
        spec = _coerce_query_spec({"question_kind": "unknown"}, question="Q")
        assert spec["question_kind"] == "factual"

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

    def test_encyclopedia_report_not_needed(self):
        raw = {"question_kind": "encyclopedia", "report_needed": False}
        spec = _coerce_query_spec(raw, question="What is GDP?")
        assert spec["report_needed"] is False
        assert spec["question_kind"] == "encyclopedia"


class TestGetTargetLangs:
    def test_empty_defaults_to_empty_list(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("QUERY_VARIANTS_LANGS", raising=False)
        assert _get_target_langs() == []

    def test_parses_comma_separated(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QUERY_VARIANTS_LANGS", "en, zh-Hans , ,fr")
        assert _get_target_langs() == ["en", "zh-Hans", "fr"]

