from application.rag_inference.deepsearch.service_bench import _filter_bench_plan_steps


def test_filter_bench_plan_steps_keeps_context_rewriter_and_skips_external() -> None:
    steps = [
        {"step_id": "plan_01", "channel": "text", "tool": "graph.context_rewriter"},
        {"step_id": "plan_02", "channel": "web", "tool": "web.search"},
        {"step_id": "plan_03", "channel": "graph", "tool": "graph.neighbors", "requires_external": True},
        {"step_id": "plan_04", "channel": "graph", "tool": "graph_adapter.query"},
    ]
    filtered = _filter_bench_plan_steps(steps)
    tool_names = {str(step.get("tool")) for step in filtered}
    assert "graph.context_rewriter" in tool_names
    assert "graph_adapter.query" in tool_names
    assert "web.search" not in tool_names
    assert "graph.neighbors" not in tool_names

