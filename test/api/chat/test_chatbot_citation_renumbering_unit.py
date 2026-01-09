import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


def test_filter_and_renumber_sources_sorted_by_original_key():
    import api.routers.chatbot as chatbot_router

    sources = [
        chatbot_router.ChatbotSourceItem(key=1, chunk_id="c1", title="t1", description="d1"),
        chatbot_router.ChatbotSourceItem(key=2, chunk_id="c2", title="t2", description="d2"),
        chatbot_router.ChatbotSourceItem(key=3, chunk_id="c3", title="t3", description="d3"),
        chatbot_router.ChatbotSourceItem(key=4, chunk_id="c4", title="t4", description="d4"),
        chatbot_router.ChatbotSourceItem(key=5, chunk_id="c5", title="t5", description="d5"),
    ]

    answer = "x<sup>1</sup> y<sup>3</sup> z<sup>4</sup>"
    updated_answer, cited_sources, key_map = chatbot_router._filter_and_renumber_sources_by_sup_keys_sorted(
        sources,
        answer,
    )

    assert updated_answer == "x<sup>1</sup> y<sup>2</sup> z<sup>3</sup>"
    assert key_map == {1: 1, 3: 2, 4: 3}
    assert [s.key for s in cited_sources] == [1, 2, 3]
    assert [s.chunk_id for s in cited_sources] == ["c1", "c3", "c4"]

