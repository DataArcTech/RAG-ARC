from core.deepsearch.report.sup_citations import convert_bracket_citations_to_sup


def test_convert_bracket_citations_to_sup_emits_single_number_tags():
    markdown = (
        "Claim A is supported [ev1]. Claim B is supported [ev2, ev3]. "
        "A markdown link should remain intact: [OpenAI](https://example.com)."
    )
    citations = [
        {"evidence_id": "ev1", "source": "graph", "used_for": "Claim A"},
        {"evidence_id": "ev2", "source": "graph", "used_for": "Claim B"},
        {"evidence_id": "ev3", "source": "graph", "used_for": "Claim B"},
    ]
    evidences = [
        {"chunk_id": "ev1", "source": "graph", "provenance": {"path": "doc1.md"}},
        {"chunk_id": "ev2", "source": "graph", "provenance": {"path": "doc2.md"}},
        {"chunk_id": "ev3", "source": "graph", "provenance": {"path": "doc3.md"}},
    ]

    converted, refs = convert_bracket_citations_to_sup(markdown, citations=citations, evidences=evidences)

    assert "<sup>1</sup>" in converted
    assert "<sup>2</sup><sup>3</sup>" in converted
    assert "<sup>1,2</sup>" not in converted
    assert "## References" in converted
    assert "[OpenAI](https://example.com)" in converted
    assert refs and refs[0]["n"] == 1 and refs[0]["evidence_id"] == "ev1"


def test_convert_bracket_citations_to_sup_does_not_touch_appendix_sections():
    markdown = "\n".join(
        [
            "Body cites [ev1].",
            "",
            "## Appendix: Chunk Evidence",
            "- [ev1] (graph): full id should remain in appendix.",
        ]
    )
    citations = [{"evidence_id": "ev1", "source": "graph"}]
    evidences = [{"chunk_id": "ev1", "source": "graph", "provenance": {"path": "doc1.md"}}]

    converted, _ = convert_bracket_citations_to_sup(markdown, citations=citations, evidences=evidences)

    assert "Body cites <sup>1</sup>." in converted
    assert "## Appendix: Chunk Evidence" in converted
    assert "- [ev1] (graph): full id should remain in appendix." in converted


def test_convert_bracket_citations_to_sup_supports_cjk_brackets():
    markdown = "结论来自证据【ev1】；补充来自【ev2】。"
    citations = [
        {"evidence_id": "ev1", "source": "graph", "used_for": "结论"},
        {"evidence_id": "ev2", "source": "graph", "used_for": "补充"},
    ]
    evidences = [
        {"chunk_id": "ev1", "source": "graph", "provenance": {"path": "doc1.md"}},
        {"chunk_id": "ev2", "source": "graph", "provenance": {"path": "doc2.md"}},
    ]

    converted, refs = convert_bracket_citations_to_sup(markdown, citations=citations, evidences=evidences)

    assert "【ev1】" not in converted
    assert "【ev2】" not in converted
    assert "<sup>1</sup>" in converted
    assert "<sup>2</sup>" in converted
    assert "## References" in converted
    assert refs and refs[0]["n"] == 1 and refs[0]["evidence_id"] == "ev1"
