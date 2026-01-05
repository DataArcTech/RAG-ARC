"""Regression suite for deterministic graph tools.

The tools under test are Neo4j-Cypher-backed deterministic operators exposed to DeepSearch
via the built-in tool catalog.
"""

import pytest

from core.deepsearch.tools import (
    GraphAggregateTool,
    GraphExpandTermsTool,
    GraphFactsByTypeTool,
    GraphIntersectionTool,
    GraphLatestTruthTool,
    GraphNeighborsTool,
    GraphPathExistsTool,
    GraphRuleCheckTool,
    GraphSdfChildrenTool,
    GraphSdfDependenciesTool,
    GraphSetDifferenceTool,
    GraphTraceToRootTool,
    ToolRunRequest,
)
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata
from core.knowledge_graph.schema import schema_from_dict


class _StubCypherAdapter:
    def __init__(self, rows_by_tool: dict[str, list[dict]], *, direction_sensitive_relations: list[str] | None = None):
        self._rows_by_tool = rows_by_tool
        self._metadata = GraphAdapterMetadata(
            adapter_name="hipporag",
            graph_type="hipporag",
            version="test",
            capabilities=(GraphAdapterCapability(name="cypher_query", modes=("neo4j",)),),
        )
        if direction_sensitive_relations:
            schema = schema_from_dict(
                {
                    "version": "v1",
                    "default_domain": "default",
                    "domains": {"default": {"direction_sensitive_relations": direction_sensitive_relations}},
                }
            )

            class _GraphStore:
                kg_schema = schema

            class _Retriever:
                graph_store = _GraphStore()

            self.retriever = _Retriever()

    async def prepare(self, question: str, *, access_scope=None) -> None:  # noqa: ARG002
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):  # noqa: ARG002
        return {"chunks": [], "nodes": [], "edges": [], "metadata": {"adapter": "hipporag"}}

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):  # noqa: ARG002
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):  # noqa: ARG002
        return "ok"

    async def chain_traverse(self, strategy, *, access_scope=None):  # noqa: ARG002
        return {"strategy": strategy.get("strategy"), "hops": 1, "visited": []}

    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ARG002
        text = str(cypher or "")
        if "AS left_fact_ids" in text and "AS right_fact_ids" in text:
            return self._rows_by_tool.get("intersection", [])
        if "set_difference" in text or "WITH u, count(rel) AS hit_count" in text:
            return self._rows_by_tool.get("set_difference", [])
        if "count(DISTINCT COALESCE(n.entity_canonical_key" in text:
            return self._rows_by_tool.get("aggregate", [])
        if "AS head_candidates" in text and "AS tail_candidates" in text and "r.fact_id AS fact_id" in text:
            head = (params or {}).get("head")
            head_key = f"rule_check:{head}"
            if head_key in self._rows_by_tool:
                return self._rows_by_tool.get(head_key, [])
            return self._rows_by_tool.get("rule_check_fail", [])
        if "AS source_candidates" in text and "AS target_candidates" in text and "AS nodes" in text:
            return self._rows_by_tool.get("path_exists", [])
        if "AS leaf_candidates" in text and "AS chain" in text:
            return self._rows_by_tool.get("trace_to_root", [])
        if "n.entity_name AS neighbor" in text and "AS candidate_count" in text:
            return self._rows_by_tool.get("neighbors", [])
        if "facts_by_type" in text or "RETURN e.entity_name AS head" in text:
            return self._rows_by_tool.get("facts_by_type", [])
        if "t.entity_name AS term" in text and "candidate_count" in text:
            return self._rows_by_tool.get("expand_terms", [])
        if "latest_truth" in text or "ORDER BY sort_key DESC" in text:
            return self._rows_by_tool.get("latest_truth", [])
        if "SDF_HAS_SUBEVENT" in text and "candidate_count AS candidate_count" in text and "c.sdf_event_id AS child_event_id" in text:
            return self._rows_by_tool.get("sdf_children", [])
        if "before_list AS before" in text and "SDF_BEFORE" in text:
            return self._rows_by_tool.get("sdf_dependencies", [])
        return []

    def metadata(self):
        return self._metadata


@pytest.mark.asyncio
async def test_example_07_intersection_query_ddi() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "intersection": [
                {
                    "target": "cyp3a4",
                    "left_fact_ids": ["fact-1"],
                    "right_fact_ids": ["fact-2"],
                    "left_source_chunk_ids": [["chunk-a"]],
                    "right_source_chunk_ids": [["chunk-b"]],
                }
            ]
        }
    )
    tool = GraphIntersectionTool()
    req = ToolRunRequest(
        question="ddi?",
        plan_step="p7",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"left": "Zenthorax", "right": "Vira-X", "left_predicates": ["INHIBITS"], "right_predicates": ["METABOLIZED_BY"]},
    )
    result = await tool.run(req)
    assert "found" in result.summary.lower()
    assert result.evidences


@pytest.mark.asyncio
async def test_example_11_set_difference_safe_products() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={"set_difference": [{"entity": "可可喜悦"}]})
    tool = GraphSetDifferenceTool()
    req = ToolRunRequest(
        question="which products do not contain peanuts?",
        plan_step="p11",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"universe_type": "Product", "exclude": ["花生"], "predicates": ["CONTAINS", "TRACES_OF"]},
    )
    result = await tool.run(req)
    assert "kept" in result.summary.lower()
    assert result.evidences


@pytest.mark.asyncio
async def test_example_12_aggregate_counts_distinct() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={"aggregate": [{"distinct_count": 3, "examples": ["apex", "beta-tech"]}]})
    tool = GraphAggregateTool()
    req = ToolRunRequest(
        question="how many suppliers?",
        plan_step="p12",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity": "Project Zeus", "predicate": "HAS_SUPPLIER", "direction": "out"},
    )
    result = await tool.run(req)
    assert "distinct_count=3" in result.summary
    assert result.evidences


@pytest.mark.asyncio
async def test_example_10_rule_check_reports_failure() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "rule_check:84消毒液": [{"fact_id": "fact-1", "source_chunk_ids": ["chunk-a"], "text": "(84,has,次氯酸钠)"}],
            "rule_check:洁厕灵": [],
        }
    )
    tool = GraphRuleCheckTool()
    req = ToolRunRequest(
        question="dangerous mix?",
        plan_step="p10",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={
            "conditions": [
                {"head": "84消毒液", "predicate": "HAS_INGREDIENT", "tail": "次氯酸钠", "direction": "out"},
                {"head": "洁厕灵", "predicate": "HAS_INGREDIENT", "tail": "盐酸", "direction": "out"},
            ]
        },
    )
    result = await tool.run(req)
    assert "failed" in result.summary.lower()
    assert result.diagnostics.get("ok") is False


@pytest.mark.asyncio
async def test_example_10_rule_check_reports_success() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "rule_check:84消毒液": [{"fact_id": "fact-1", "source_chunk_ids": ["chunk-a"], "text": "(84,has,次氯酸钠)"}],
            "rule_check:洁厕灵": [{"fact_id": "fact-2", "source_chunk_ids": ["chunk-b"], "text": "(洁厕灵,has,盐酸)"}],
        }
    )
    tool = GraphRuleCheckTool()
    req = ToolRunRequest(
        question="dangerous mix?",
        plan_step="p10-ok",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={
            "conditions": [
                {"head": "84消毒液", "predicate": "HAS_INGREDIENT", "tail": "次氯酸钠", "direction": "out"},
                {"head": "洁厕灵", "predicate": "HAS_INGREDIENT", "tail": "盐酸", "direction": "out"},
            ]
        },
    )
    result = await tool.run(req)
    assert "passed" in result.summary.lower()
    assert result.diagnostics.get("ok") is True


@pytest.mark.asyncio
async def test_rule_check_fails_when_no_valid_conditions() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={})
    tool = GraphRuleCheckTool()
    req = ToolRunRequest(
        question="noop should fail",
        plan_step="p10-noop",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"conditions": [{}, {"head": "", "predicate": "", "tail": ""}]},
    )
    result = await tool.run(req)
    assert "failed" in result.summary.lower()
    assert result.diagnostics.get("ok") is False


@pytest.mark.asyncio
async def test_example_01_multi_hop_path_exists() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "path_exists": [
                {
                    "nodes": ["a公司", "b公司", "c公司"],
                    "predicates": ["OWNS", "OWNS"],
                    "fact_ids": ["fact-1", "fact-2"],
                    "source_chunk_ids": [["chunk-1"], ["chunk-2"]],
                }
            ]
        }
    )
    tool = GraphPathExistsTool()
    req = ToolRunRequest(
        question="A 是否间接控股 C？",
        plan_step="p01",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"source": "A公司", "target": "C公司", "predicates": ["OWNS"], "direction": "out", "max_hops": 3},
    )
    result = await tool.run(req)
    assert "ok=true" in result.summary.lower()
    assert result.evidences


@pytest.mark.asyncio
async def test_sdf_children_returns_children() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "sdf_children": [
                {
                    "candidate_count": 1,
                    "gate": "and",
                    "child": "生效期校验",
                    "child_event_id": "sdf-ev-b",
                    "importance": 1.0,
                    "source_chunk_ids": ["chunk-a"],
                }
            ]
        }
    )
    tool = GraphSdfChildrenTool()
    req = ToolRunRequest(
        question="保险责任裁决流程是什么？",
        plan_step="p-sdf-children",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"event": "保险责任裁决", "doc_namespace": "doc-1", "limit": 10},
    )
    result = await tool.run(req)
    assert "children=1" in result.summary
    assert result.evidences
    assert result.evidences[0].provenance.get("children")


@pytest.mark.asyncio
async def test_sdf_dependencies_returns_neighbors() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "sdf_dependencies": [
                {
                    "candidate_count": 1,
                    "before": [{"name": "生效期校验", "event_id": "sdf-ev-b", "source_chunk_ids": ["chunk-a"]}],
                    "after": [{"name": "除外条款校验", "event_id": "sdf-ev-c", "source_chunk_ids": ["chunk-b"]}],
                }
            ]
        }
    )
    tool = GraphSdfDependenciesTool()
    req = ToolRunRequest(
        question="哪些步骤在除外条款校验之前/之后？",
        plan_step="p-sdf-deps",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"event": "除外条款校验", "doc_namespace": "doc-1", "limit": 10},
    )
    result = await tool.run(req)
    assert "before=1" in result.summary
    assert "after=1" in result.summary
    assert result.evidences


@pytest.mark.asyncio
async def test_direction_sensitive_predicate_forces_directed_traversal() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "path_exists": [
                {
                    "nodes": ["a公司", "b公司", "c公司"],
                    "predicates": ["OWNS", "OWNS"],
                    "fact_ids": ["fact-1", "fact-2"],
                    "source_chunk_ids": [["chunk-1"], ["chunk-2"]],
                }
            ]
        },
        direction_sensitive_relations=["OWNS"],
    )
    tool = GraphPathExistsTool()
    req = ToolRunRequest(
        question="A 是否间接控股 C？",
        plan_step="p01-both",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"source": "A公司", "target": "C公司", "predicates": ["OWNS"], "direction": "both", "max_hops": 3},
    )
    result = await tool.run(req)
    assert result.evidences
    assert result.evidences[0].provenance.get("direction") == "out"
    assert result.evidences[0].provenance.get("direction_forced_sensitive") is True
    assert result.evidences[0].provenance.get("direction_forced_undirected") is False


@pytest.mark.asyncio
async def test_example_02_entity_disambiguation_filter_by_type() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "facts_by_type": [
                {
                    "head": "捷豹路虎(公司)",
                    "predicate": "HAS_FACT",
                    "tail": "Q3营收增长12%",
                    "fact_id": "fact-1",
                    "source_chunk_ids": [["chunk-1"]],
                }
            ]
        }
    )
    tool = GraphFactsByTypeTool()
    req = ToolRunRequest(
        question="捷豹在第三季度表现如何？",
        plan_step="p02",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity_type": "Company", "predicates": ["HAS_FACT"], "limit": 10},
    )
    result = await tool.run(req)
    assert "type=Company" in result.summary
    assert result.evidences
    facts = (result.diagnostics or {}).get("facts") or []
    assert any(item.get("tail") == "Q3营收增长12%" for item in facts)


def test_example_03_schema_hallucination_rejects_unknown_relations() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {
                "default": {
                    "allowed_relations": ["PARTNERS_WITH", "SUPPLIES", "DEVELOPS", "ACQUIRED"],
                    "relation_aliases": {"supplies": "SUPPLIES"},
                    "unknown_predicate_policy": "reject",
                }
            },
        }
    )
    domain = schema.for_domain("default")
    assert domain.normalize_predicate("SUPPLIES") == "SUPPLIES"
    assert domain.normalize_predicate("DISCUSSED_WITH") is None


@pytest.mark.asyncio
async def test_example_04_directionality_neighbors_in_out() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "neighbors": [
                {"neighbor": "Novacorp", "predicate": "OWNS", "fact_id": "fact-1", "source_chunk_ids": [["chunk-1"]]},
                {"neighbor": "TinyAI", "predicate": "OWNS", "fact_id": "fact-2", "source_chunk_ids": [["chunk-2"]]},
            ]
        }
    )
    tool = GraphNeighborsTool()
    # predecessors (in)
    req_in = ToolRunRequest(
        question="谁拥有 Stratos Global？",
        plan_step="p04-in",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity": "Stratos Global", "predicates": ["OWNS"], "direction": "in", "limit": 10},
    )
    result_in = await tool.run(req_in)
    assert "direction=in" in result_in.summary
    # successors (out)
    req_out = ToolRunRequest(
        question="Stratos Global 拥有谁？",
        plan_step="p04-out",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity": "Stratos Global", "predicates": ["OWNS"], "direction": "out", "limit": 10},
    )
    result_out = await tool.run(req_out)
    assert "direction=out" in result_out.summary


@pytest.mark.asyncio
async def test_example_05_fragmented_evidence_list_all_features() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "neighbors": [
                {"neighbor": "钛合金机身", "predicate": "HAS_FEATURE", "fact_id": "f1", "source_chunk_ids": [["c1"]]},
                {"neighbor": "浪涌保护", "predicate": "HAS_FEATURE", "fact_id": "f2", "source_chunk_ids": [["c2"]]},
                {"neighbor": "激光雷达避障", "predicate": "HAS_FEATURE", "fact_id": "f3", "source_chunk_ids": [["c3"]]},
                {"neighbor": "红色急停按钮", "predicate": "HAS_FEATURE", "fact_id": "f4", "source_chunk_ids": [["c4"]]},
                {"neighbor": "ISO-9001 认证", "predicate": "HAS_FEATURE", "fact_id": "f5", "source_chunk_ids": [["c5"]]},
            ]
        }
    )
    tool = GraphNeighborsTool()
    req = ToolRunRequest(
        question="列出 Model-X 的全部特性",
        plan_step="p05",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity": "Model-X", "predicates": ["HAS_FEATURE"], "direction": "out", "limit": 20},
    )
    result = await tool.run(req)
    neighbors = (result.diagnostics or {}).get("neighbors") or []
    assert len(neighbors) == 5


@pytest.mark.asyncio
async def test_example_06_hierarchy_trace_to_root_returns_chain() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "trace_to_root": [
                {"chain": ["Zeus-X 超级计算机", "核心处理单元", "量子芯片组", "Qubit Lattice"], "hops": 3}
            ]
        }
    )
    tool = GraphTraceToRootTool()
    req = ToolRunRequest(
        question="追溯 Qubit Lattice 的层级路径",
        plan_step="p06",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"leaf": "Qubit Lattice", "predicates": ["CONTAINS"], "max_hops": 6},
    )
    result = await tool.run(req)
    chain = (result.diagnostics or {}).get("chain") or []
    assert chain == ["Zeus-X 超级计算机", "核心处理单元", "量子芯片组", "Qubit Lattice"]


@pytest.mark.asyncio
async def test_example_08_ontology_mapping_query_expansion() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={"expand_terms": [{"term": "Cart-Flow-V2"}]})
    tool = GraphExpandTermsTool()
    req = ToolRunRequest(
        question="结账服务错误率怎么样？",
        plan_step="p08",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"concept": "结账服务", "predicates": ["IMPLEMENTS"], "direction": "in", "limit": 10},
    )
    result = await tool.run(req)
    terms = (result.diagnostics or {}).get("terms") or []
    assert "Cart-Flow-V2" in terms


@pytest.mark.asyncio
async def test_example_09_temporal_conflict_latest_truth() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "latest_truth": [
                {
                    "value": "0 天/周",
                    "predicate": "HAS_POLICY",
                    "sort_key": "2024",
                    "fact_id": "fact-2024",
                    "source_chunk_ids": [["chunk-2024"]],
                }
            ]
        }
    )
    tool = GraphLatestTruthTool()
    req = ToolRunRequest(
        question="每周可远程办公几天？",
        plan_step="p09",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"topic": "远程办公政策", "predicates": ["HAS_POLICY"]},
    )
    result = await tool.run(req)
    assert "value=0 天/周" in result.summary


def test_example_12_entity_canonicalization_configurable() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {
                "default": {
                    "entity_suffixes_to_strip": ["corp", "inc", "ltd", "llc"],
                    "entity_aliases": {"gamma logistics": "gamma logistics"},
                }
            },
        }
    )
    domain = schema.for_domain("default")
    assert domain.canonicalize_entity_name("Apex Corp") == "apex"
    assert domain.canonicalize_entity_name("Apex Inc") == "apex"
