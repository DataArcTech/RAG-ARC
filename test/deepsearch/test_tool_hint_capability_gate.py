import pytest

from core.deepsearch.tooling import describe_available_tools
from core.deepsearch.tooling.adapter_capability_gate import disabled_tools_for_adapter, merge_disabled_tools
from core.deepsearch.tooling.registry import ToolHintRegistry
from core.graph_adapter.base import GraphAdapterCapability, GraphAdapterMetadata


def _tool_names(include_llm_tools: bool, *, registry: ToolHintRegistry) -> set[str]:
    return {hint.get("name") for hint in describe_available_tools(registry=registry, include_llm_tools=include_llm_tools)}


def test_missing_cypher_capability_hides_cypher_tools() -> None:
    metadata = GraphAdapterMetadata(
        adapter_name="stub",
        graph_type="stub",
        version="v1",
        capabilities=(GraphAdapterCapability(name="chain_of_exploration", modes=("ppr_chain", "ppr_prefetch", "bridge_lookup")),),
    )
    registry = ToolHintRegistry()
    disabled = disabled_tools_for_adapter(metadata)
    registry.set_disabled_tools(disabled)

    names = _tool_names(include_llm_tools=True, registry=registry)
    assert "graph.intersection" not in names
    assert "graph.aggregate" not in names
    assert "graph.path_exists" not in names


def test_chain_mode_gates_only_missing_modes() -> None:
    metadata = GraphAdapterMetadata(
        adapter_name="stub",
        graph_type="stub",
        version="v1",
        capabilities=(
            GraphAdapterCapability(name="cypher_query", modes=("neo4j",)),
            GraphAdapterCapability(name="chain_of_exploration", modes=("ppr_chain", "ppr_prefetch", "bridge_lookup")),
        ),
    )
    registry = ToolHintRegistry()
    disabled = disabled_tools_for_adapter(metadata)
    registry.set_disabled_tools(merge_disabled_tools(disabled, []))

    names = _tool_names(include_llm_tools=True, registry=registry)
    assert "graph.bridge_lookup" in names
    assert "graph.path_cache" in names
    assert "graph.beam_search" not in names
    assert "graph.intersection" in names

