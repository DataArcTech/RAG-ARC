from encapsulation.data_model.deepsearch import EvidenceChunk

from core.deepsearch.report import DeepSearchReporter


def _build_trace(include_final_answer: bool = True):
    trace = {
        "question": "Who partnered with OpenAI?",
        "plan_steps": [
            {"step_id": "plan_01", "description": "Collect graph facts", "channel": "graph", "metadata": {}},
        ],
        "reasoning_steps": [
            {
                "step_id": "plan_01",
                "description": "Collect graph facts",
                "channel": "graph",
                "status": "done",
                "output_summary": "Microsoft formed a strategic partnership with OpenAI in 2019.",
                "produced_evidence_ids": ["ev1"],
            }
        ],
        "graph_traversals": [
            {
                "visited_nodes": ["OpenAI", "Microsoft"],
                "visited_edges": ["OpenAI-works_with-Microsoft"],
            }
        ],
        "evidences": [
            {
                "chunk_id": "ev1",
                "source": "hipporag",
                "content": "OpenAI entered a partnership with Microsoft in 2019.",
            }
        ],
        "adapter_metadata": {"adapter_name": "hipporag"},
        "coverage_metrics": {"coverage_ratio": 0.75},
        "graph_context": {
            "adapter_name": "hipporag",
            "question": "Who partnered with OpenAI?",
            "metadata": {
                "request_metadata": {
                    "conversation_id": "conv-123",
                    "locale": "en-US",
                }
            },
        },
    }
    if include_final_answer:
        trace["final_answer"] = "Microsoft maintains the flagship partnership with OpenAI, enabling Azure-hosted deployments."
    return trace


def test_reporter_prefers_final_answer_and_merges_evidence():
    reporter = DeepSearchReporter(template_store=None, config={"parallel_thinking_runs": 2})
    trace = _build_trace(include_final_answer=True)
    external = [EvidenceChunk(chunk_id="ev2", source="press", content="Azure hosts OpenAI services.")]

    report = reporter.compose(trace, external)

    assert report["answer"].startswith("## Final Answer"), "answer should include contextual heading"
    assert "Microsoft maintains the flagship partnership" in report["answer"], "Final answer content should persist"
    assert "Microsoft maintains the flagship partnership" in report["structured_report"]["summary"]
    assert len(report["evidences"]) == 2
    assert report["metadata"]["plan"]["completed"] == 1
    assert report["metadata"]["graph_summary"]["unique_nodes"] == 2
    assert report["metadata"]["parallel_thinking_runs"] == 2
    assert report["metadata"]["request_context"]["conversation_id"] == "conv-123"


def test_reporter_builds_highlights_when_final_answer_missing():
    reporter = DeepSearchReporter(template_store={DeepSearchReporter.ANSWER_TEMPLATE_KEY: "Graph findings:"}, config={})
    trace = _build_trace(include_final_answer=False)

    report = reporter.compose(trace, external_evidence=[])

    assert report["structured_report"]["summary"].startswith("Graph findings:"), "Template override should apply"
    assert report["highlights"], "High-level summaries should be extracted from reasoning steps"
    assert "Evidence collected" not in report["structured_report"]["summary"], "Highlights should populate summary"
