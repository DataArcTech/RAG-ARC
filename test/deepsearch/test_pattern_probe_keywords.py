from core.deepsearch.tools.fast.pattern_probe import PatternProbeTool


def test_pattern_probe_picks_domain_terms_without_jieba(monkeypatch):
    tool = PatternProbeTool(max_terms=4)
    monkeypatch.setattr(tool, "_jieba_tokens", lambda _text: [])
    question = "请对比港险储蓄/年金产品：供款年期、保证回报/非保证回报、提取/退保规则、身故保障与风险提示。"
    keywords = tool._pick_keywords(question, extra={})
    assert keywords
    assert any(term in "".join(keywords) for term in ("供款", "退保", "回报", "身故", "年金"))

