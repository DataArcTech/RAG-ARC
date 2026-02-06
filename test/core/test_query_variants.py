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


def test_query_variants_deterministic_variants_work_without_llm(monkeypatch):
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_ENABLED", True)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LANGS", ["zh-Hant", "en"])
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_MAX", 5)
    monkeypatch.setattr(query_variants, "_opencc_convert", lambda base, config: "計劃特點" if config == "s2t" else None)

    variants = query_variants.generate_query_variants("计划特点 C508A", llm_connector=None)
    assert variants[0] == "计划特点 C508A"
    assert "計劃特點" in variants
    assert "C508A" in variants
