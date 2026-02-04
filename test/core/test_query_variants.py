from core.utils import query_variants


def test_query_variants_always_include_original(monkeypatch):
    monkeypatch.setattr(
        query_variants,
        "_llm_rewrite_variants",
        lambda _llm, base, langs: {lang: f"{base}-{lang}" for lang in langs},
    )
    assert query_variants.generate_query_variants("  hello  ", llm_connector=object())[0] == "hello"


def test_query_variants_generate_lang_variants(monkeypatch):
    monkeypatch.setattr(
        query_variants,
        "_llm_rewrite_variants",
        lambda _llm, base, langs: {"zh-Hans": base, "zh-Hant": "計劃特點", "en": "plan features"},
    )
    variants = query_variants.generate_query_variants("计划特点", llm_connector=object())
    assert variants[0] == "计划特点"
    assert "計劃特點" in variants
    assert "plan features" in variants
    assert len(variants) == len(set(variants))


def test_query_variants_keep_entity_tokens(monkeypatch):
    monkeypatch.setattr(
        query_variants,
        "_llm_rewrite_variants",
        lambda _llm, base, langs: {"en": "China Life C508A-C516A 2026_01 Zhiyu"},
    )
    variants = query_variants.generate_query_variants("中国人寿 C508A-C516A 2026_01 智裕世代", llm_connector=object())
    assert variants[0] == "中国人寿 C508A-C516A 2026_01 智裕世代"
    assert "China Life C508A-C516A 2026_01 Zhiyu" in variants
