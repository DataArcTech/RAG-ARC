from encapsulation.web_search.aggregation import aggregate_tavily_results
from encapsulation.web_search.tavily_client import TavilySearchResult


def test_aggregate_tavily_results_groups_by_domain():
    results = [
        TavilySearchResult(
            title="A1",
            url="https://a.example/x",
            content="alpha",
            score=0.9,
        ),
        TavilySearchResult(
            title="A2",
            url="https://a.example/y",
            content="beta",
            score=0.8,
        ),
        TavilySearchResult(
            title="B1",
            url="https://b.example/z",
            content="gamma",
            score=0.95,
        ),
    ]

    out, stats = aggregate_tavily_results(
        results,
        group_by="domain",
        max_groups=1,
        max_items_per_group=2,
    )

    assert len(out) == 2
    assert [item.url for item in out] == ["https://a.example/x", "https://a.example/y"]
    assert stats.total_in == 3
    assert stats.total_out == 2
    assert stats.groups and stats.groups[0]["key"] == "a.example"
