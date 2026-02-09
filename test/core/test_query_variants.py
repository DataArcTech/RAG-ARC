from core.utils import query_variants


def test_query_variants_always_include_original(monkeypatch):
    monkeypatch.setattr(
        query_variants,
        "_llm_rewrite_variants",
        lambda _llm, base, langs, *, cache_scope=None: {lang: f"{base}-{lang}" for lang in langs},
    )
    assert query_variants.generate_query_variants("  hello  ", llm_connector=object())[0] == "hello"


def test_query_variants_generate_lang_variants(monkeypatch):
    monkeypatch.setattr(
        query_variants,
        "_llm_rewrite_variants",
        lambda _llm, base, langs, *, cache_scope=None: {"zh-Hans": base, "zh-Hant": "計劃特點", "en": "plan features"},
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
        lambda _llm, base, langs, *, cache_scope=None: {"en": "China Life C508A-C516A 2026_01 Zhiyu"},
    )
    variants = query_variants.generate_query_variants("中国人寿 C508A-C516A 2026_01 智裕世代", llm_connector=object())
    assert variants[0] == "中国人寿 C508A-C516A 2026_01 智裕世代"
    assert "China Life C508A-C516A 2026_01 Zhiyu" in variants


def test_query_variants_llm_cache_is_scoped(monkeypatch):
    # Ensure the in-process LLM rewrite cache never cross-contaminates between scopes.
    # We use a deterministic stub connector and count chat calls.
    class _Cfg:
        model_name = "stub-model"
        low_cost_model_name = None

    class _StubLLM:
        config = _Cfg()

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, _messages, **_kwargs):
            self.calls += 1
            return '{"zh-Hans":"A","zh-Hant":"B","en":"C"}'

    llm = _StubLLM()
    monkeypatch.setattr(query_variants, "_LLM_REWRITE_CACHE", None)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_ENABLED", True)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LLM_CACHE_ENABLED", True)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LLM_CACHE_MAX_ENTRIES", 128)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LLM_CACHE_TTL_SECONDS", 3600)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LLM_MAX_TOKENS", 64)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LANGS", ["zh-Hans", "zh-Hant", "en"])

    _ = query_variants.generate_query_variants("计划特点", llm_connector=llm, cache_scope="owner_a")
    _ = query_variants.generate_query_variants("计划特点", llm_connector=llm, cache_scope="owner_a")
    assert llm.calls == 1

    _ = query_variants.generate_query_variants("计划特点", llm_connector=llm, cache_scope="owner_b")
    assert llm.calls == 2

    # No scope => caching disabled (safe default).
    _ = query_variants.generate_query_variants("计划特点", llm_connector=llm, cache_scope=None)
    _ = query_variants.generate_query_variants("计划特点", llm_connector=llm, cache_scope=None)
    assert llm.calls == 4


def test_query_variants_deterministic_variants_work_without_llm(monkeypatch):
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_ENABLED", True)
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_LANGS", ["zh-Hant", "en"])
    monkeypatch.setattr(query_variants, "QUERY_VARIANTS_MAX", 5)
    monkeypatch.setattr(query_variants, "_opencc_convert", lambda base, config: "計劃特點" if config == "s2t" else None)

    variants = query_variants.generate_query_variants("计划特点 C508A", llm_connector=None)
    assert variants[0] == "计划特点 C508A"
    assert "計劃特點" in variants
    assert "C508A" in variants
