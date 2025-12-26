from core.deepsearch.report.composer import _split_authoritative_evidences


def test_report_filters_tool_generated_evidence_from_authoritative_bundle():
    evidences = [
        {
            "chunk_id": "graph.context_rollup:plan_01:rollup:abc123",
            "source": "context_rollup",
            "content": "LLM rollup output that should not be citable.",
        },
        {
            "chunk_id": "1a9a49a1-8193-4608-8938-250e62731697",
            "source": "hipporag",
            "content": "Authoritative corpus chunk.",
        },
        {
            "chunk_id": "ext-1",
            "source": "web.stub",
            "content": "External evidence chunk.",
        },
    ]

    authoritative, tool_generated = _split_authoritative_evidences(evidences)

    assert [ev["chunk_id"] for ev in authoritative] == [
        "1a9a49a1-8193-4608-8938-250e62731697",
        "ext-1",
    ]
    assert [ev["chunk_id"] for ev in tool_generated] == ["graph.context_rollup:plan_01:rollup:abc123"]

