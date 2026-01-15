from core.deepsearch.reasoning.multi_agent import MultiAgentGraphReasoningLoop, MultiAgentSettings


def _settings(*, min_evidence: int, min_cov: float) -> MultiAgentSettings:
    return MultiAgentSettings(
        enabled=True,
        max_subagents=1,
        subagent_concurrency=1,
        enable_parallel_tool_probes=False,
        probe_tool_names=(),
        probe_concurrency=1,
        lead_tool_names=(),
        lead_tool_concurrency=1,
        worker_timeout_seconds=None,
        worker_retry_attempts=0,
        fail_fast=False,
        incremental_parallelism=True,
        initial_worker_count=1,
        stop_min_evidence_count=min_evidence,
        stop_min_coverage_ratio=min_cov,
        max_merge_evidences=50,
    )


def test_multi_agent_stop_counts_primary_evidence_only():
    settings = _settings(min_evidence=3, min_cov=0.9)

    merged = {
        "evidences": [
            {"chunk_id": "d1", "source": "graph.context_rollup", "content": "x", "kind": "derived"},
            {"chunk_id": "x1", "source": "graph.neighbors", "content": "x", "kind": "diagnostic"},
            {"chunk_id": "d2", "source": "graph.context_rollup", "content": "x", "kind": "derived"},
        ],
        "coverage_metrics": {"coverage_ratio": 1.0},
    }
    assert MultiAgentGraphReasoningLoop._should_stop_incremental(merged, settings=settings) is False

    merged["evidences"].extend(
        [
            {"chunk_id": "c1", "source": "graph_adapter.query", "content": "x", "kind": "primary"},
            {"chunk_id": "c2", "source": "graph_adapter.query", "content": "x", "kind": "primary"},
            {"chunk_id": "c3", "source": "graph_adapter.query", "content": "x", "kind": "primary"},
        ]
    )
    assert MultiAgentGraphReasoningLoop._should_stop_incremental(merged, settings=settings) is True

