from core.deepsearch.tooling.adapter_capability_gate import chain_mode_required_tool_modes, cypher_required_tool_names
from core.deepsearch.tools import builtin_tool_descriptors, llm_optional_tool_names, llm_required_tool_names
from core.deepsearch.tools.governance_tags import (
    EVIDENCE_DERIVED,
    EVIDENCE_PRIMARY,
    REQUIRES_CHAIN_TRAVERSE,
    REQUIRES_CYPHER,
    REQUIRES_LLM,
    SCOPE_FILE,
    SCOPE_OWNER,
)


def test_tool_descriptors_have_unique_names_and_namespaces() -> None:
    descriptors = list(builtin_tool_descriptors())
    assert descriptors

    names = [d.name for d in descriptors]
    assert len(names) == len(set(names))

    mcp_names = [d.namespace for d in descriptors if d.mcp_callable]
    assert mcp_names
    assert all(isinstance(ns, str) and ns.strip() and "." in ns for ns in mcp_names)
    assert len(mcp_names) == len(set(mcp_names))


def test_mcp_specs_are_well_formed() -> None:
    for desc in builtin_tool_descriptors():
        spec = desc.as_mcp_spec()
        assert isinstance(spec, dict)
        assert spec.get("name") == desc.namespace
        assert isinstance(spec.get("description"), str) and spec["description"].strip()
        schema = spec.get("input_schema")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"


def test_tool_descriptors_follow_governance_contract() -> None:
    descriptors = list(builtin_tool_descriptors())
    assert descriptors

    for desc in descriptors:
        tags = set(desc.strategy_tags or [])
        assert SCOPE_OWNER in tags, f"{desc.name} must declare owner-scope semantics"

        evidence_tags = {EVIDENCE_PRIMARY, EVIDENCE_DERIVED} & tags
        assert len(evidence_tags) == 1, f"{desc.name} must declare exactly one evidence tag"

        if SCOPE_FILE in tags:
            assert desc.channel in {"graph", "text"}

        if desc.mcp_callable:
            assert isinstance(desc.example_args, dict) and desc.example_args, f"{desc.name} must provide example_args"
            spec = desc.as_mcp_spec()
            examples = spec.get("examples")
            assert isinstance(examples, list) and examples, f"{desc.name} must expose MCP examples"


def test_requirement_tags_match_runtime_gates() -> None:
    descriptor_by_name = {desc.name: desc for desc in builtin_tool_descriptors()}

    for tool_name in cypher_required_tool_names():
        desc = descriptor_by_name.get(tool_name)
        assert desc is not None, f"cypher-required tool missing from catalog: {tool_name}"
        assert REQUIRES_CYPHER in set(desc.strategy_tags or []), f"{tool_name} must declare {REQUIRES_CYPHER}"

    for tool_name in chain_mode_required_tool_modes().keys():
        desc = descriptor_by_name.get(tool_name)
        assert desc is not None, f"chain-mode tool missing from catalog: {tool_name}"
        assert REQUIRES_CHAIN_TRAVERSE in set(desc.strategy_tags or []), f"{tool_name} must declare {REQUIRES_CHAIN_TRAVERSE}"

    for tool_name in (llm_required_tool_names() | llm_optional_tool_names()):
        desc = descriptor_by_name.get(tool_name)
        assert desc is not None, f"LLM tool missing from catalog: {tool_name}"
        assert REQUIRES_LLM in set(desc.strategy_tags or []), f"{tool_name} must declare {REQUIRES_LLM}"


def test_tool_catalog_excludes_deprecated_tools() -> None:
    names = {desc.name for desc in builtin_tool_descriptors()}
    assert "graph.multi_adapter_compare" not in names
