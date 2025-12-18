from api.routers.chatbot import ChatbotSourceItem, _extract_sup_keys, _filter_sources_by_sup_keys


def test_extract_sup_keys_dedup_preserves_order():
    assert _extract_sup_keys("A<sup>2</sup> B<sup>1</sup> C<sup>2</sup>") == [2, 1]


def test_filter_sources_by_sup_keys_empty_when_no_citation():
    sources = [
        ChatbotSourceItem(key=1, title="a", description="d"),
        ChatbotSourceItem(key=2, title="b", description="d"),
    ]
    assert _filter_sources_by_sup_keys(sources, "hello") == []


def test_filter_sources_by_sup_keys_keeps_only_referenced():
    sources = [
        ChatbotSourceItem(key=1, title="a", description="d"),
        ChatbotSourceItem(key=2, title="b", description="d"),
        ChatbotSourceItem(key=3, title="c", description="d"),
    ]
    kept = _filter_sources_by_sup_keys(sources, "x<sup>3</sup>y<sup>1</sup>")
    assert [s.key for s in kept] == [1, 3]

