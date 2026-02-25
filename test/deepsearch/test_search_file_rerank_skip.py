from core.deepsearch.tools import LocateTool


def test_search_file_rerank_skip_triggers_on_margin_when_not_blocked() -> None:
    skip, diag = LocateTool._should_skip_rerank(  # type: ignore[attr-defined]
        query="pricing overview",
        candidates=[{"score": 1.5}, {"score": 1.0}],
    )
    assert skip is True
    assert diag.get("reason") == "confident_top1"


def test_search_file_rerank_skip_blocks_on_entity_attribute_query_cue() -> None:
    skip, diag = LocateTool._should_skip_rerank(  # type: ignore[attr-defined]
        query="请问A公司产品a1是否有特点b？",
        candidates=[{"score": 10.0}, {"score": 1.0}],
    )
    assert skip is False
    assert diag.get("reason") == "blocked_by_query_cue"


def test_search_file_rerank_skip_does_not_trigger_when_margin_small() -> None:
    skip, diag = LocateTool._should_skip_rerank(  # type: ignore[attr-defined]
        query="pricing overview",
        candidates=[{"score": 1.05}, {"score": 1.0}],
    )
    assert skip is False
    assert diag.get("reason") == "margin_too_small"
