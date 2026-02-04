from core.deepsearch.report.composer import _split_authoritative_evidences
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_GRAPH_INFERENCE


def test_report_filters_tool_generated_evidence_from_authoritative_bundle():
    evidences = [
        {
            "chunk_id": "graph.llm_chain_explorer:plan_01:rollup:abc123",
            "source": "graph.llm_chain_explorer",
            "content": "LLM rollup output that should not be citable.",
        },
        {
            "chunk_id": "read.pages:1",
            "source": "read.pages",
            "content": "Authoritative page evidence.",
        },
        {
            "chunk_id": "graph.ops:plan_01:path:xyz",
            "source": "graph.ops",
            "content": "A->rel->B",
            "provenance": {"evidence_class": EVIDENCE_CLASS_GRAPH_INFERENCE},
        },
        {
            "chunk_id": "ext-1",
            "source": "web.stub",
            "content": "External evidence chunk.",
        },
    ]

    authoritative, tool_generated = _split_authoritative_evidences(evidences)

    assert [ev["chunk_id"] for ev in authoritative] == ["read.pages:1", "graph.ops:plan_01:path:xyz"]
    assert {ev["chunk_id"] for ev in tool_generated} == {
        "graph.llm_chain_explorer:plan_01:rollup:abc123",
        "ext-1",
    }
