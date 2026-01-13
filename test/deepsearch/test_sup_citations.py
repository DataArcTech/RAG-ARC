from core.deepsearch.report.sup_citations import convert_bracket_citations_to_sup


def test_convert_bracket_citations_to_sup_emits_contiguous_sup_tags_and_sources():
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
        {
            "chunk_id": "ev1",
            "source": "hipporag",
            "content": "Chunk 1 content",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-1", "filename": "doc1.md"}}},
        },
        {
            "chunk_id": "ev2",
            "source": "web.tavily",
            "content": "Web title line\nWeb snippet body",
            "provenance": {"url": "https://example.com/doc2"},
        },
        {
            "chunk_id": "ev3",
            "source": "hipporag",
            "content": "Chunk 3 content",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-3", "filename": "doc3.md"}}},
        },
    ]

    converted, sources, citation_key_map = convert_bracket_citations_to_sup(markdown, citations=citations, evidences=evidences)

    assert "<sup>1</sup>" in converted
    assert "<sup>2</sup><sup>3</sup>" in converted
    assert "## Evidence Index" not in converted
    assert "[OpenAI](https://example.com)" in converted
    assert citation_key_map == {"ev1": 1, "ev2": 2, "ev3": 3}
    assert sources and [entry["key"] for entry in sources] == [1, 2, 3]
    assert [entry["chunk_id"] for entry in sources] == ["ev1", "ev2", "ev3"]
    assert sources[0]["file"] == "/knowledge/file-1/download"
    assert sources[0]["title"] == "doc1.md"
    assert sources[1]["file"] == "https://example.com/doc2"
    assert sources[1]["title"] == "Web title line"
    assert sources[2]["file"] == "/knowledge/file-3/download"
    assert sources[2]["title"] == "doc3.md"


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
    evidences = [
        {
            "chunk_id": "ev1",
            "source": "hipporag",
            "content": "Chunk 1 content",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-1", "filename": "doc1.md"}}},
        }
    ]

    converted, _, _ = convert_bracket_citations_to_sup(markdown, citations=citations, evidences=evidences)

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
        {
            "chunk_id": "ev1",
            "source": "hipporag",
            "content": "Chunk 1 content",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-1", "filename": "doc1.md"}}},
        },
        {
            "chunk_id": "ev2",
            "source": "hipporag",
            "content": "Chunk 2 content",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-2", "filename": "doc2.md"}}},
        },
    ]

    converted, sources, citation_key_map = convert_bracket_citations_to_sup(markdown, citations=citations, evidences=evidences)

    assert "【ev1】" not in converted
    assert "【ev2】" not in converted
    assert "<sup>1</sup>" in converted
    assert "<sup>2</sup>" in converted
    assert "## Evidence Index" not in converted
    assert citation_key_map == {"ev1": 1, "ev2": 2}
    assert sources and sources[0]["key"] == 1 and sources[0]["chunk_id"] == "ev1"


def test_convert_bracket_citations_to_sup_drops_unused_citations_from_sources():
    markdown = "Only this claim is cited [ev1]."
    citations = [
        {"evidence_id": "ev1", "source": "graph"},
        {"evidence_id": "ev2", "source": "graph"},
    ]
    evidences = [
        {
            "chunk_id": "ev1",
            "source": "hipporag",
            "content": "Chunk 1 content",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-1", "filename": "doc1.md"}}},
        },
        {
            "chunk_id": "ev2",
            "source": "hipporag",
            "content": "Chunk 2 content",
            "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-2", "filename": "doc2.md"}}},
        },
    ]

    converted, sources, citation_key_map = convert_bracket_citations_to_sup(markdown, citations=citations, evidences=evidences)

    assert "<sup>1</sup>" in converted
    assert "ev2" not in converted
    assert citation_key_map == {"ev1": 1}
    assert sources and [entry["chunk_id"] for entry in sources] == ["ev1"]
