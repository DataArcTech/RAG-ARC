"""Adapter capability gating for DeepSearch tool hints.

DeepSearch ships with a mixed tool catalog:
- deterministic tools backed by a graph database (e.g. Neo4j Cypher)
- traversal/cache helpers backed by adapter chain-of-exploration primitives
- LLM-heavy tools that may depend on adapter traversal outputs

To avoid "configured/enabled but not actually supported" breakage, we compute a
disabled-tool set from adapter metadata and feed it into ToolHintRegistry so
planner prompts only advertise tools that are expected to work.
"""
from typing import Iterable, Mapping, Set

from core.graph_adapter.base import GraphAdapterMetadata


_CYPHER_REQUIRED_TOOLS: Set[str] = {
    "graph.intersection",
    "graph.set_difference",
    "graph.aggregate",
    "graph.entity_concepts",
    "graph.rule_check",
    "graph.schema_nodes",
    "graph.path_exists",
    "graph.neighbors",
    "graph.facts_by_type",
    "graph.expand_terms",
    "graph.latest_truth",
    "graph.sdf_children",
    "graph.sdf_dependencies",
    "graph.trace_to_root",
}

_CHAIN_MODE_REQUIRED_TOOLS: Mapping[str, str] = {
    "graph.bridge_lookup": "bridge_lookup",
    "graph.path_cache": "ppr_prefetch",
    "graph.beam_search": "beam_search",
}


def _capability_index(metadata: GraphAdapterMetadata) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for cap in getattr(metadata, "capabilities", ()) or ():
        name = getattr(cap, "name", None)
        if not name:
            continue
        modes_raw = getattr(cap, "modes", ()) or ()
        if isinstance(modes_raw, (list, tuple, set)):
            modes = {str(mode) for mode in modes_raw if str(mode).strip()}
        else:
            modes = {str(modes_raw)} if str(modes_raw).strip() else set()
        index[str(name)] = modes
    return index


def disabled_tools_for_adapter(metadata: GraphAdapterMetadata) -> set[str]:
    """Return tool names that should be hidden for a given adapter."""

    caps = _capability_index(metadata)
    disabled: set[str] = set()

    if "cypher_query" not in caps:
        disabled.update(_CYPHER_REQUIRED_TOOLS)

    chain_modes = caps.get("chain_of_exploration", set())
    for tool_name, required_mode in _CHAIN_MODE_REQUIRED_TOOLS.items():
        if required_mode not in chain_modes:
            disabled.add(tool_name)

    return disabled


def merge_disabled_tools(*groups: Iterable[str]) -> set[str]:
    merged: set[str] = set()
    for group in groups:
        if not group:
            continue
        for name in group:
            token = str(name).strip()
            if token:
                merged.add(token)
    return merged
