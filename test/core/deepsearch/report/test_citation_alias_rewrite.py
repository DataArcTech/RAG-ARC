import pytest


def test_rewrite_citation_aliases_rewrites_source_key_map() -> None:
    from core.deepsearch.report.composer import DeepSearchReporter

    structured = {
        "text": "Answer with citations: <sup>1</sup> <sup>2</sup>",
        "source_key_map": {"1": "chunk_001", 2: "chunk_002"},
        "citations": [{"evidence_id": "chunk_001"}, {"evidence_id": "chunk_002"}],
    }
    alias_bundle = {
        "alias_to_original": {
            "chunk_001": "real-chunk-a",
            "chunk_002": "real-chunk-b",
        }
    }

    # We only need the pure rewrite behavior; no LLM/graph config.
    rewritten, diag = DeepSearchReporter._rewrite_citation_aliases(  # type: ignore[attr-defined]
        DeepSearchReporter,  # unbound "self" is fine; method is pure for our usage
        structured,
        alias_bundle,
    )

    assert rewritten["source_key_map"]["1"] == "real-chunk-a"
    assert rewritten["source_key_map"]["2"] == "real-chunk-b"
    assert rewritten["citations"][0]["evidence_id"] == "real-chunk-a"
    assert rewritten["citations"][1]["evidence_id"] == "real-chunk-b"
    assert diag.get("replaced", 0) >= 2


@pytest.mark.parametrize(
    "raw_val, expected",
    [
        ("chunk_1", "real-chunk-a"),
        ("CHUNK-001", "real-chunk-a"),
        ("  chunk_001  ", "real-chunk-a"),
    ],
)
def test_rewrite_citation_aliases_normalizes_alias_variants(raw_val: str, expected: str) -> None:
    from core.deepsearch.report.composer import DeepSearchReporter

    structured = {"text": "", "source_key_map": {"1": raw_val}}
    alias_bundle = {"alias_to_original": {"chunk_001": "real-chunk-a"}}
    rewritten, _diag = DeepSearchReporter._rewrite_citation_aliases(  # type: ignore[attr-defined]
        DeepSearchReporter,
        structured,
        alias_bundle,
    )
    assert rewritten["source_key_map"]["1"] == expected

