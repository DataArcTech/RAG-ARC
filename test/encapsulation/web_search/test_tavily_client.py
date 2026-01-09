from encapsulation.web_search.tavily_client import TavilySearchClient


def test_tavily_client_coerces_results_and_formats_evidence_chunks():
    payload = {
        "results": [
            {"title": "A", "url": "https://a.example", "content": "alpha", "score": 0.9},
            {"title": "B", "url": "https://b.example", "content": "beta", "score": 0.1},
            {"title": "C", "url": None, "content": "", "score": 0.5},  # dropped (empty content)
            "bad",  # dropped
        ]
    }

    results = TavilySearchClient._coerce_results(payload)
    assert [r.title for r in results] == ["A", "B"]
    assert results[0].url == "https://a.example"

    evidences = TavilySearchClient.to_evidence_chunks(results=results, step_id="s1", query="q")
    assert len(evidences) == 2
    assert evidences[0]["chunk_id"] == "tavily-s1-0"
    assert evidences[0]["source"] == "web.tavily"
    assert evidences[0]["provenance"]["provider"] == "tavily"
    assert evidences[0]["provenance"]["url"] == "https://a.example"

