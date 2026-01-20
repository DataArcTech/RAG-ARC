from application.rag_inference.deepsearch.service_bench import _filter_bench_plan_steps


def test_filter_bench_plan_steps_keeps_llm_chain_explorer_and_skips_external() -> None:
    steps = [
        {"step_id": "plan_01", "channel": "text", "tool": "graph.llm_chain_explorer"},
        {"step_id": "plan_02", "channel": "web", "tool": "web.search"},
        {"step_id": "plan_03", "channel": "graph", "tool": "graph.neighbors"},
        {"step_id": "plan_04", "channel": "graph", "tool": "graph_adapter.query"},
    ]
    filtered = _filter_bench_plan_steps(steps)
    tool_names = {str(step.get("tool")) for step in filtered}
    assert "graph.llm_chain_explorer" in tool_names
    assert "graph_adapter.query" in tool_names
    assert "web.search" not in tool_names
    assert "graph.neighbors" in tool_names
