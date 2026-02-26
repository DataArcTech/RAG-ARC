from application.rag_inference.deepsearch.service_runtime.artifact_dedupe_v2 import (
    build_evidence_pool_v2,
    dedupe_reasoning_v2,
    dedupe_report_v2,
)


def test_deepsearch_artifact_dedupe_v2_builds_evidence_pool_and_refs():
    reasoning = {
        "plan_steps": [{"step_id": "s1"}],
        "evidences": [{"chunk_id": "c1", "text": "t1"}, {"chunk_id": "c2", "text": "t2"}],
        "tool_results": [
            {
                "tool_name": "locate",
                "result": {"evidences": [{"chunk_id": "c1", "text": "t1"}]},
            }
        ],
        "reasoning_steps": [{"status": "done"}],
        "think_notes": [{"note": "x"}],
        "coverage_metrics": {"coverage_ratio": 0.5},
        "graph_traversals": [{"node": "n1"}],
    }

    report = {
        "question": "q",
        "answer": "final answer",
        "evidences": [{"chunk_id": "c2", "text": "t2"}, {"chunk_id": "c3", "text": "t3"}],
        "structured_report": {"format_version": 1, "text": "final answer"},
        "metadata": {
            "tool_results": reasoning["tool_results"],
            "reasoning_steps": reasoning["reasoning_steps"],
            "think_notes": reasoning["think_notes"],
            "coverage_metrics": reasoning["coverage_metrics"],
            "graph_visualization": reasoning["graph_traversals"],
            "plan": {"steps": [{"step_id": "s1"}]},
            "structured_report": {"text": "final answer"},
        },
    }

    pool, reasoning_ids, report_ids = build_evidence_pool_v2(reasoning=reasoning, report=report, artifact_version=2)
    assert pool["kind"] == "evidence_pool"
    assert pool["order"] == ["c1", "c2", "c3"]
    assert set(pool["evidences_by_id"]) == {"c1", "c2", "c3"}
    assert reasoning_ids == ["c1", "c2"]
    assert report_ids == ["c2", "c3"]

    deduped_reasoning = dedupe_reasoning_v2(
        reasoning=reasoning,
        refs_enabled=True,
        evidence_pool_filename="evidence_pool.json",
        plan_filename="plan_result.json",
        evidence_ids=reasoning_ids,
    )
    assert "evidences" not in deduped_reasoning
    assert "plan_steps" not in deduped_reasoning
    assert deduped_reasoning["plan_ref"]["$ref"] == {"file": "plan_result.json", "json_pointer": "/plan"}
    assert deduped_reasoning["evidence_pool_ref"]["$ref"]["file"] == "evidence_pool.json"
    tool_result = deduped_reasoning["tool_results"][0]["result"]
    assert "evidences" not in tool_result
    assert tool_result["evidence_ids"] == ["c1"]

    deduped_report = dedupe_report_v2(
        report=report,
        refs_enabled=True,
        report_markdown_filename="report.md",
        evidence_pool_filename="evidence_pool.json",
        reasoning_filename="reasoning.json",
        plan_filename="plan_result.json",
        evidence_ids=report_ids,
    )
    assert "answer" not in deduped_report
    assert deduped_report["answer_ref"]["$ref"]["file"] == "report.md"
    assert "evidences" not in deduped_report
    assert deduped_report["evidence_pool_ref"]["$ref"]["file"] == "evidence_pool.json"
    assert deduped_report["evidence_ids"] == ["c2", "c3"]
    assert "text" not in deduped_report["structured_report"]
    assert deduped_report["structured_report"]["text_ref"]["$ref"]["file"] == "report.md"

    meta = deduped_report["metadata"]
    assert "tool_results" not in meta
    assert meta["tool_results_ref"]["$ref"] == {"file": "reasoning.json", "json_pointer": "/tool_results"}
    assert meta["reasoning_steps_ref"]["$ref"] == {"file": "reasoning.json", "json_pointer": "/reasoning_steps"}
    assert meta["think_notes_ref"]["$ref"] == {"file": "reasoning.json", "json_pointer": "/think_notes"}
    assert meta["coverage_metrics_ref"]["$ref"] == {"file": "reasoning.json", "json_pointer": "/coverage_metrics"}
    assert meta["graph_visualization_ref"]["$ref"] == {"file": "reasoning.json", "json_pointer": "/graph_traversals"}
    assert meta["plan_ref"]["$ref"] == {"file": "plan_result.json", "json_pointer": "/plan"}
    assert meta["reasoning_ref"]["$ref"]["file"] == "reasoning.json"


def test_deepsearch_artifact_dedupe_v2_prefers_structured_report_evidence_ids():
    reasoning = {
        "evidences": [{"chunk_id": "c1", "text": "t1"}],
        "tool_results": [
            {
                "tool_name": "read.pages",
                "result": {"evidences": [{"chunk_id": "p1", "text": "page evidence"}]},
            }
        ],
    }
    # report.evidences may include navigation snippets; structured_report.source_key_map tells what is citeable.
    report = {
        "answer": "final answer <sup>1</sup>",
        "evidences": [{"chunk_id": "bm25-1", "text": "snippet"}, {"chunk_id": "p1", "text": "page evidence"}],
        "structured_report": {"source_key_map": {"1": "p1"}},
    }

    pool, _reasoning_ids, report_ids = build_evidence_pool_v2(reasoning=reasoning, report=report, artifact_version=2)
    assert pool["kind"] == "evidence_pool"
    assert "p1" in pool["evidences_by_id"]
    # Only include citeable evidence ids for report (avoid listing navigation snippets as report evidence).
    assert report_ids == ["p1"]
